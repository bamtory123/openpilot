import json
import os
import signal
import threading
import functools
import time
import numpy as np
from openpilot.common.transformations.camera import DEVICE_CAMERAS, get_view_frame_from_calib_frame

from collections import namedtuple
from enum import Enum
from multiprocessing import Event, Process, Queue, Value
from abc import ABC, abstractmethod
from opendbc.car.honda.values import CruiseButtons
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.test.helpers import set_params_enabled
from openpilot.tools.sim.lib.common import SimulatorState, World
from openpilot.tools.sim.lib.simulated_car import SimulatedCar
from openpilot.tools.sim.lib.simulated_sensors import SimulatedSensors

QueueMessage = namedtuple("QueueMessage", ["type", "info"], defaults=[None])

class QueueMessageType(Enum):
  START_STATUS = 0
  CONTROL_COMMAND = 1
  TERMINATION_INFO = 2
  CLOSE_STATUS = 3
  TELEMETRY = 4

def control_cmd_gen(cmd: str):
  return QueueMessage(QueueMessageType.CONTROL_COMMAND, cmd)

def rk_loop(function, hz, exit_event: threading.Event):
  rk = Ratekeeper(hz, None)
  while not exit_event.is_set():
    function()
    rk.keep_time()


def build_model_debug_snapshot(model, calibration, camera_config, requested_camera_source_frame_id: int,
                               bridge_frame: int) -> dict:
  def points(value):
    return [[float(x), float(y), float(z)] for x, y, z in zip(value.x, value.y, value.z, strict=True)]

  rpy = list(calibration.rpyCalib) if len(calibration.rpyCalib) == 3 else [0.0, 0.0, 0.0]
  height = list(calibration.height)
  camera = camera_config.narrow_road
  return {
    "schema_version": 1,
    "scope": "analysis_only_model_projection_not_runtime_control_or_accuracy",
    "requested_camera_source_frame_id": requested_camera_source_frame_id,
    "bridge_frame": bridge_frame,
    "model_frame_id": int(model.frameId),
    "projection": {
      "intrinsic": camera.intrinsics.tolist(),
      "view_from_calib": get_view_frame_from_calib_frame(*rpy, 0.0)[:, :3].tolist(),
      "camera_height_m": float(height[0]) if height else 1.22,
      "calibration_rpy_rad": [float(value) for value in rpy],
    },
    "path": points(model.position),
    "lane_lines": [points(lane) for lane in model.laneLines],
    "lane_line_probabilities": [float(value) for value in model.laneLineProbs],
  }


class SimulatorBridge(ABC):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, simlab_config=None):
    set_params_enabled()
    self.params = Params()
    self.params.put_bool("AlphaLongitudinalEnabled", True, block=True)

    self.rk = Ratekeeper(100, None)

    self.dual_camera = dual_camera
    self.high_quality = high_quality
    self.simlab_config = simlab_config or {}
    capture_frames = os.environ.get("SIMLAB_MODEL_DEBUG_SOURCE_FRAME_IDS", "")
    self.model_debug_capture_frames = [int(value) for value in capture_frames.split(",") if value]
    self.model_debug_capture_dir = os.environ.get("SIMLAB_MODEL_DEBUG_CAPTURE_DIR")
    if self.model_debug_capture_frames and not self.model_debug_capture_dir:
      raise ValueError("SIMLAB_MODEL_DEBUG_CAPTURE_DIR is required when model source frames are configured")
    self.model_debug_capture_index = 0

    self._exit_event: threading.Event | None = None
    self._threads = []
    self._keep_alive = True
    self._shutdown_event = Event()
    self.started = Value('i', False)
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self.simulator_state = SimulatorState()

    self.world: World | None = None

    self.past_startup_engaged = False
    self.startup_button_prev = True

    self.test_run = False

  def _on_shutdown(self, signal, frame):
    self.shutdown()

  def shutdown(self):
    self._keep_alive = False
    self._shutdown_event.set()

  def bridge_keep_alive(self, control_q: Queue, status_q: Queue, retries: int):
    try:
      self._run(control_q, status_q)
    finally:
      self.close("bridge terminated")

  def close(self, reason):
    self.started.value = False

    if self._exit_event is not None:
      self._exit_event.set()

    if self.world is not None:
      self.world.close(reason)

  def run(self, control_queue, retries=-1, status_queue=None):
    # Keep the historical single-queue API by default, but allow experiment
    # runners to prevent the bridge from consuming its own telemetry.
    status_queue = control_queue if status_queue is None else status_queue
    bridge_p = Process(name="bridge", target=self.bridge_keep_alive, args=(control_queue, status_queue, retries))
    bridge_p.start()
    return bridge_p

  def print_status(self):
    print(
    f"""
State:
Ignition: {self.simulator_state.ignition} Engaged: {self.simulator_state.is_engaged}
    """)

  @abstractmethod
  def spawn_world(self, q: Queue, /) -> World:
    pass

  def _run(self, control_q: Queue, status_q: Queue):
    self.world = self.spawn_world(status_q)

    self.simulated_car = SimulatedCar()
    camera_fault = self.simlab_config.get("fault", {})
    camera_sink = getattr(self.world, "emit_camera_telemetry", None)
    self.simulated_sensors = SimulatedSensors(self.dual_camera, camera_fault, camera_sink)

    self._exit_event = threading.Event()

    self.simulated_car_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_car.update, self.simulator_state),
                                                                        100, self._exit_event))
    self.simulated_car_thread.start()

    self.simulated_camera_thread = threading.Thread(target=rk_loop, args=(functools.partial(self.simulated_sensors.send_camera_images, self.world),
                                                                        20, self._exit_event))
    self.simulated_camera_thread.start()

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      self.world.tick()

    fault_enabled = False
    measurement_announced = False
    fault_enabled_at = None
    fault_settle_s = float(self.simlab_config.get("run", {}).get("fault_settle_s", 5))
    while self._keep_alive and not self._shutdown_event.is_set():
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0

      self.simulator_state.cruise_button = 0
      self.simulator_state.left_blinker = False
      self.simulator_state.right_blinker = False

      throttle_manual = steer_manual = brake_manual = 0.

      # Read manual controls
      if not control_q.empty():
        message = control_q.get()
        if message.type == QueueMessageType.CONTROL_COMMAND:
          m = message.info.split('_')
          if m[0] == "steer":
            steer_manual = float(m[1])
          elif m[0] == "throttle":
            throttle_manual = float(m[1])
          elif m[0] == "brake":
            brake_manual = float(m[1])
          elif m[0] == "cruise":
            if m[1] == "down":
              self.simulator_state.cruise_button = CruiseButtons.DECEL_SET
            elif m[1] == "up":
              self.simulator_state.cruise_button = CruiseButtons.RES_ACCEL
            elif m[1] == "cancel":
              self.simulator_state.cruise_button = CruiseButtons.CANCEL
            elif m[1] == "main":
              self.simulator_state.cruise_button = CruiseButtons.MAIN
          elif m[0] == "blinker":
            if m[1] == "left":
              self.simulator_state.left_blinker = True
            elif m[1] == "right":
              self.simulator_state.right_blinker = True
          elif m[0] == "ignition":
            self.simulator_state.ignition = not self.simulator_state.ignition
          elif m[0] == "reset":
            self.world.reset()
          elif m[0] == "quit":
            break

      self.simulator_state.user_brake = brake_manual
      self.simulator_state.user_gas = throttle_manual
      self.simulator_state.user_torque = steer_manual * -10000

      steer_manual = steer_manual * -40

      # Update openpilot on current sensor state
      self.simulated_sensors.update(self.simulator_state, self.world)

      self.simulated_car.sm.update(0)
      self.simulator_state.is_engaged = self.simulated_car.sm['selfdriveState'].active

      if self.rk.frame % 10 == 0:
        selfdrive_state = self.simulated_car.sm['selfdriveState']
        status_q.put(QueueMessage(QueueMessageType.TELEMETRY, {
          "type": "openpilot_state", "frame": self.rk.frame,
          "engageable": bool(selfdrive_state.engageable),
          "engaged": bool(self.simulator_state.is_engaged),
          "state": str(selfdrive_state.state), "alert_text_1": selfdrive_state.alertText1,
          "alert_text_2": selfdrive_state.alertText2,
          "onroad_events": [str(event.name) for event in self.simulated_car.sm['onroadEvents']],
        }))

      if self.simulator_state.is_engaged:
        throttle_op = np.clip(self.simulated_car.sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = np.clip(-self.simulated_car.sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = self.simulated_car.sm['carControl'].actuators.steeringAngleDeg

        self.past_startup_engaged = True
      elif not self.past_startup_engaged and self.simulated_car.sm['selfdriveState'].engageable:
        self.simulator_state.cruise_button = CruiseButtons.DECEL_SET if self.startup_button_prev else CruiseButtons.MAIN # force engagement on startup
        self.startup_button_prev = not self.startup_button_prev

      throttle_out = throttle_op if self.simulator_state.is_engaged else throttle_manual
      brake_out = brake_op if self.simulator_state.is_engaged else brake_manual
      steer_out = steer_op if self.simulator_state.is_engaged else steer_manual

      if hasattr(self.world, "set_control_telemetry"):
        model = self.simulated_car.sm['modelV2']
        model_curvature = model.action.desiredCurvature
        planner_curvature = self.simulated_car.sm['lateralManeuverPlan'].desiredCurvature
        control_curvature = self.simulated_car.sm['controlsState'].desiredCurvature
        path_x = np.asarray(model.position.x)
        path_y = np.asarray(model.position.y)
        path_valid = len(path_x) == len(path_y) and len(path_x) > 1
        path_reaches_20m = path_valid and path_x[-1] >= 20.0
        path_y_20m = float(np.interp(20.0, path_x, path_y)) if path_reaches_20m else None
        path_heading_20m = float(np.arctan2(path_y_20m - path_y[0], 20.0 - path_x[0])) if path_reaches_20m else None
        path_end_x = float(path_x[-1]) if path_valid else 0.0
        path_end_y = float(path_y[-1]) if path_valid else 0.0
        path_end_heading = float(np.arctan2(path_y[-1] - path_y[0], path_x[-1] - path_x[0])) if path_valid else 0.0
        path_end_speed = float(model.velocity.x[-1]) if len(model.velocity.x) else None
        calibration = self.simulated_car.sm['extrinsicsCalibration']
        calibration_rpy = list(calibration.rpyCalib) if len(calibration.rpyCalib) == 3 else [None, None, None]
        device_type = str(self.simulated_car.sm['deviceState'].deviceType)
        camera_sensor = str(self.simulated_car.sm['narrowRoadCameraState'].sensor)
        camera_config = DEVICE_CAMERAS.get((device_type, camera_sensor))
        if (camera_config is not None and self.model_debug_capture_index < len(self.model_debug_capture_frames)
            and model.frameId >= self.model_debug_capture_frames[self.model_debug_capture_index]):
          requested_frame_id = self.model_debug_capture_frames[self.model_debug_capture_index]
          snapshot = build_model_debug_snapshot(model, calibration, camera_config, requested_frame_id, self.rk.frame)
          capture_dir = os.path.abspath(self.model_debug_capture_dir)
          os.makedirs(capture_dir, exist_ok=True)
          with open(os.path.join(capture_dir, f"model-source-frame-{model.frameId:06d}.json"), "w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2, sort_keys=True)
            stream.write("\n")
          self.model_debug_capture_index += 1
        left_lane = model.laneLines[1] if len(model.laneLines) > 1 else None
        right_lane = model.laneLines[2] if len(model.laneLines) > 2 else None
        left_lane_prob = model.laneLineProbs[1] if len(model.laneLineProbs) > 1 else None
        right_lane_prob = model.laneLineProbs[2] if len(model.laneLineProbs) > 2 else None
        left_lane_y0 = left_lane.y[0] if left_lane and len(left_lane.y) else None
        right_lane_y0 = right_lane.y[0] if right_lane and len(right_lane.y) else None
        self.world.set_control_telemetry(steer_op, self.simulated_car.sm['carControl'].actuators.accel if self.simulator_state.is_engaged else 0.0,
                                         throttle_out, brake_out, model_curvature, planner_curvature, control_curvature,
                                         path_y_20m, path_heading_20m, path_end_x, path_end_y, path_end_heading, path_end_speed,
                                         self.simulated_car.sm.valid['modelV2'], model.frameId, model.frameAge,
                                         model.frameDropPerc, model.modelExecutionTime, calibration_rpy, str(calibration.calStatus),
                                         device_type, camera_sensor, camera_config.narrow_road.width if camera_config else None,
                                         camera_config.narrow_road.height if camera_config else None,
                                         camera_config.narrow_road.focal_length if camera_config else None,
                                         left_lane_prob, right_lane_prob, left_lane_y0, right_lane_y0)

      if self.simulator_state.is_engaged and not fault_enabled:
        self.simulated_sensors.enable_camera_transport_fault(True)
        fault_enabled = True
        fault_enabled_at = time.monotonic()
        if hasattr(self.world, "emit_run_event"):
          self.world.emit_run_event("ENABLE_FAULT")
      if fault_enabled and not measurement_announced and fault_enabled_at is not None and time.monotonic() - fault_enabled_at >= fault_settle_s:
        measurement_announced = True
        if hasattr(self.world, "emit_run_event"):
          self.world.emit_run_event("MEASURE")

      self.world.apply_controls(steer_out, throttle_out, brake_out)
      self.world.read_state()
      self.world.read_sensors(self.simulator_state)

      if self.world.exit_event.is_set():
        self.shutdown()

      if self.rk.frame % self.TICKS_PER_FRAME == 0:
        self.world.tick()
        self.world.read_cameras()

      # don't print during test, so no print/IO Block between OP and metadrive processes
      if not self.test_run and self.rk.frame % 25 == 0:
        self.print_status()

      self.started.value = True

      self.rk.keep_time()

    self.simulated_sensors.close()
