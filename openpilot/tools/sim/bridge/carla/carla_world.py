from __future__ import annotations

import math
from pathlib import Path
from queue import Empty, Queue
import time
from typing import Any

import numpy as np

from openpilot.tools.sim.bridge.carla.route_asset import load_route_asset
from openpilot.tools.sim.bridge.carla.control import normalized_steer
from openpilot.tools.sim.bridge.carla.optics import narrow_road_fov_deg
from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.lib.camerad import H, W
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3


HONDA_CIVIC_2022_STEER_RATIO = 15.38
# C3 narrow-road camera focal length (2648 px at the 1928 px simulator width).
# CARLA defaults to 90°, which presents a materially different road geometry.
NARROW_ROAD_FOV_DEG = narrow_road_fov_deg(W)
ROUTE_DEPARTURE_M = 1.5


class CarlaWorld(World):
  """Synchronous CARLA implementation of the existing ``World`` contract.

  The route asset selects a reproducible start pose only. It is diagnostic data,
  never an input to steering or longitudinal control.
  """

  def __init__(self, status_q, *, dual_camera: bool, host: str, port: int, town: str,
               route_asset: str, test_duration: float, test_run: bool):
    super().__init__(dual_camera)
    try:
      import carla
    except ImportError as error:
      raise RuntimeError("CARLA Python API is unavailable; install carla==0.9.16") from error
    self.carla, self.status_q, self.town = carla, status_q, town
    self.test_duration, self.test_run = test_duration, test_run
    self.route = load_route_asset(Path(route_asset), expected_town=town)
    self.client = carla.Client(host, port)
    self.client.set_timeout(30.0)
    self.world = self.client.get_world()
    observed_town = self.world.get_map().name.rsplit("/", 1)[-1]
    if observed_town != town:
      raise RuntimeError(f"CARLA town mismatch: {observed_town} != {town}; launcher must provision the map")
    self.original_settings = self.world.get_settings()
    settings = self.world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    self.world.apply_settings(settings)
    self._actors: list[Any] = []
    self._sensors: list[Any] = []
    self._frames: Queue[Any] = Queue(maxsize=2)
    self._collisions: Queue[Any] = Queue()
    self._lanes: Queue[Any] = Queue()
    self._closed = False
    self._engaged_at: float | None = None
    self._last_yaw_deg: float | None = None
    self._last_frame = -1
    self._route_cursor = 0
    self._physics: dict[str, Any] = {}
    self._route_lateral_error_m: float | None = None
    self._control = carla.VehicleControl()
    self._command: dict[str, Any] = {}
    self._spawn_ego()
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, {"backend": "carla", "town": town,
                      "route_asset_sha256": self.route.sha256}))

  def _transform(self, point):
    x, y, z, pitch, yaw, roll = point
    return self.carla.Transform(self.carla.Location(x=x, y=y, z=z), self.carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

  def _put_latest(self, target: Queue[Any], value: Any):
    try:
      target.put_nowait(value)
    except Exception:
      try:
        target.get_nowait()
      except Empty:
        pass
      target.put_nowait(value)

  def _spawn_ego(self):
    blueprints = self.world.get_blueprint_library()
    vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
    spawn = self._transform(self.route.points[0])
    # Waypoints lie on the road surface; CARLA vehicle spawning needs a small
    # clearance above it. This changes neither frozen XY nor yaw alignment.
    spawn.location.z += 0.5
    self.ego = self.world.try_spawn_actor(vehicle_bp, spawn)
    if self.ego is None:
      raise RuntimeError("route asset spawn is occupied")
    # In synchronous mode CARLA does not expose the finalized actor transform
    # until the first world tick.
    self.world.tick()
    actual = self.ego.get_transform()
    xy_error = math.hypot(actual.location.x - spawn.location.x, actual.location.y - spawn.location.y)
    yaw_error = abs((actual.rotation.yaw - spawn.rotation.yaw + 180.0) % 360.0 - 180.0)
    if xy_error > 1.0 or yaw_error > 5.0:
      self.ego.destroy()
      raise RuntimeError(f"route asset spawn did not align with ego: xy={xy_error:.3f}m yaw={yaw_error:.3f}deg")
    self._actors.append(self.ego)
    limits = [wheel.max_steer_angle for wheel in self.ego.get_physics_control().wheels if wheel.max_steer_angle > 0]
    if not limits:
      raise RuntimeError("CARLA ego exposes no front steering limit")
    self._steering_wheel_limit_deg = max(limits) * HONDA_CIVIC_2022_STEER_RATIO
    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(W))
    camera_bp.set_attribute("image_size_y", str(H))
    camera_bp.set_attribute("fov", f"{NARROW_ROAD_FOV_DEG:.6f}")
    camera_bp.set_attribute("sensor_tick", "0.05")
    camera = self.world.spawn_actor(camera_bp, self.carla.Transform(self.carla.Location(x=1.45, z=1.35)), attach_to=self.ego)
    camera.listen(self._on_camera)
    self._sensors.append(camera)
    collision = self.world.spawn_actor(blueprints.find("sensor.other.collision"), self.carla.Transform(), attach_to=self.ego)
    collision.listen(self._collisions.put)
    self._sensors.append(collision)
    lane = self.world.spawn_actor(blueprints.find("sensor.other.lane_invasion"), self.carla.Transform(), attach_to=self.ego)
    lane.listen(self._lanes.put)
    self._sensors.append(lane)

  def _on_camera(self, image):
    rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))[:, :, :3][:, :, ::-1].copy()
    self._put_latest(self._frames, (image.frame, time.monotonic_ns(), rgb))

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    # openpilot desired curvature/steering-wheel convention is opposite to
    # CARLA's normalized road-wheel steering convention.
    steer = normalized_steer(-float(steer_angle), self._steering_wheel_limit_deg)
    self._control = self.carla.VehicleControl(steer=steer, throttle=float(np.clip(throttle_out, 0.0, 1.0)),
                                              brake=float(np.clip(brake_out, 0.0, 1.0)))
    self.ego.apply_control(self._control)
    self._command.update({"openpilot_steering_wheel_deg": float(steer_angle), "carla_normalized_steer": steer,
                          "carla_throttle": self._control.throttle, "carla_brake": self._control.brake,
                          "control_source": "openpilot"})

  def set_control_telemetry(self, steering_angle_deg, accel, throttle, brake, model_curvature, planner_curvature,
                            control_curvature, *unused):
    self._command.update({"openpilot_steering_wheel_deg": float(steering_angle_deg), "openpilot_accel_mps2": float(accel),
                          "model_target_curvature_1pm": float(model_curvature),
                          "planner_target_curvature_1pm": float(planner_curvature),
                          "control_target_curvature_1pm": float(control_curvature),
                          "openpilot_throttle": float(throttle), "openpilot_brake": float(brake)})

  def emit_camera_telemetry(self, payload):
    self.status_q.put(QueueMessage(QueueMessageType.TELEMETRY, payload))

  def emit_run_event(self, state):
    self.status_q.put(QueueMessage(QueueMessageType.TELEMETRY, {"type": "run_state", "state": state,
                      "backend": "carla", "mono_ns": time.monotonic_ns()}))

  def tick(self):
    self.world.tick()
    snapshot = self.world.get_snapshot()
    if snapshot.frame == self._last_frame:
      return
    self._last_frame = snapshot.frame
    transform, velocity = self.ego.get_transform(), self.ego.get_velocity()
    speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    yaw_rate = None
    if self._last_yaw_deg is not None:
      delta = (transform.rotation.yaw - self._last_yaw_deg + 180.0) % 360.0 - 180.0
      yaw_rate = math.radians(delta) / max(snapshot.timestamp.delta_seconds, 1e-3)
    self._last_yaw_deg = transform.rotation.yaw
    self._physics = {"type": "vehicle_telemetry", "backend": "carla",
      "simulation_frame": snapshot.frame, "simulation_time_s": snapshot.timestamp.elapsed_seconds,
      "speed_mps": speed, "yaw_deg": transform.rotation.yaw,
      "yaw_rate_radps": yaw_rate, "actual_curvature_1pm": yaw_rate / speed if yaw_rate is not None and speed > 0.1 else None,
      "carla_x_m": transform.location.x, "carla_y_m": transform.location.y, **self._route_ground_truth(transform)}

  def _route_ground_truth(self, transform):
    """Route-asset ground truth for diagnostics only; never control input."""
    location = transform.location
    points = self.route.points
    low, high = max(0, self._route_cursor - 8), min(len(points), self._route_cursor + 40)
    self._route_cursor = min(range(low, high), key=lambda index: math.hypot(location.x - points[index][0], location.y - points[index][1]))
    x, y, _, _, yaw, _ = points[self._route_cursor]
    yaw_rad = math.radians(yaw)
    dx, dy = location.x - x, location.y - y
    lateral_error = dx * -math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    self._route_lateral_error_m = lateral_error
    heading_error = (transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0
    before, after = max(0, self._route_cursor - 2), min(len(points) - 1, self._route_cursor + 10)
    route_curvature = None
    if after > before:
      delta_yaw = (points[after][4] - points[before][4] + 180.0) % 360.0 - 180.0
      route_curvature = math.radians(delta_yaw) / ((after - before) * 2.0)
    return {"route_progress_m": self._route_cursor * 2.0, "route_lateral_error_m": lateral_error,
            "route_heading_error_deg": heading_error, "route_reference_curvature_1pm": route_curvature}

  def read_state(self):
    if not self._collisions.empty():
      self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"collision": True}))
      self.exit_event.set()
    lane_invasions = 0
    while not self._lanes.empty():
      self._lanes.get_nowait()
      lane_invasions += 1
    if lane_invasions:
      self.status_q.put(QueueMessage(QueueMessageType.TELEMETRY, {"type": "lane_invasion", "count": lane_invasions,
                        "route_lateral_error_m": self._route_lateral_error_m}))
    if self._engaged_at is not None and self._route_lateral_error_m is not None and abs(self._route_lateral_error_m) > ROUTE_DEPARTURE_M:
      self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"lane_departure": True,
                        "route_lateral_error_m": self._route_lateral_error_m, "threshold_m": ROUTE_DEPARTURE_M}))
      self.exit_event.set()
    if self.test_run and self._engaged_at is not None and time.monotonic() - self._engaged_at >= self.test_duration:
      self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"timeout": True}))
      self.exit_event.set()

  def read_sensors(self, state: SimulatorState):
    transform, velocity = self.ego.get_transform(), self.ego.get_velocity()
    state.velocity = vec3(float(velocity.x), float(velocity.y), float(velocity.z))
    state.bearing = float(transform.rotation.yaw)
    state.imu.bearing = state.bearing
    state.imu.accelerometer = vec3(0.0, 0.0, 0.0)
    state.imu.gyroscope = vec3(0.0, 0.0, 0.0)
    state.gps.from_xy((transform.location.x, transform.location.y))
    state.steering_angle = self._control.steer * self._steering_wheel_limit_deg
    state.valid = True
    if self._physics:
      self.status_q.put(QueueMessage(QueueMessageType.TELEMETRY, {**self._physics, **self._command,
                        "mono_ns": time.monotonic_ns()}))
    if state.is_engaged and self._engaged_at is None:
      self._engaged_at = time.monotonic()

  def read_cameras(self):
    try:
      _, _, image = self._frames.get(timeout=2.0)
    except Empty as error:
      raise RuntimeError("CARLA camera did not produce a frame within 2 seconds") from error
    self.road_image[...] = image
    self.image_lock.release()

  def reset(self):
    raise RuntimeError("CARLA adapter pilot does not support in-run reset; start a new isolated run")

  def close(self, reason: str):
    if self._closed:
      return
    self._closed = True
    for sensor in reversed(self._sensors):
      try:
        sensor.stop()
        sensor.destroy()
      except RuntimeError:
        pass
    for actor in reversed(self._actors):
      try:
        actor.destroy()
      except RuntimeError:
        pass
    self.world.apply_settings(self.original_settings)
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, {"backend": "carla", "reason": reason}))
