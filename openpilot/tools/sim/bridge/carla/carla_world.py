import math
import os
import queue
import time
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from openpilot.tools.sim.bridge.common import QueueMessage, QueueMessageType
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3
from openpilot.tools.sim.lib.camerad import H, W


# C3 narrow-road optics. The mount is expressed in CARLA Tesla Model 3 local
# coordinates, not MetaDrive vehicle coordinates: x=0 places the sensor inside
# CARLA's cabin, where the dashboard and roof obstruct the learned-model view.
CAMERA_FOCAL_LENGTH_PX = 2648.0
CAMERA_HORIZONTAL_FOV_DEG = math.degrees(2.0 * math.atan(W / (2.0 * CAMERA_FOCAL_LENGTH_PX)))
CAMERA_MOUNT = (1.45, 0.0, 1.35, 0.0)  # x, y, z, pitch; CARLA windshield/road-camera position
# Honda Civic 2022 CarSpecs. CARLA's normalized steer controls road-wheel angle,
# while openpilot's steeringAngleDeg is steering-wheel angle.
HONDA_CIVIC_2022_STEER_RATIO = 15.38
SCENARIO_ENTRY_SPEED_MPS = {
  "curve_60s": 8.0,
  "straight_30": 30.0 / 3.6,
  "straight_50": 50.0 / 3.6,
  "straight_70": 70.0 / 3.6,
  "lead_follow_30": 50.0 / 3.6,
  "lead_stop_restart": 40.0 / 3.6,
  "city_mixed": 7.0,
}
LEAD_SCENARIOS = {"lead_follow_30", "lead_stop_restart"}
CITY_TURN_PLAN = ("straight", "left", "straight", "right", "straight", "u_turn", "straight")
# Measured around the straight-road calibration: normalized CARLA steer 0.02
# produces roughly 0.007 1/m. This conversion is simulator-only.
SPECIALIST_CURVATURE_PER_NORMALIZED_STEER = 0.35
SPECIALIST_MAX_NORMALIZED_STEER = 0.03
GROUND_TRUTH_MAX_NORMALIZED_STEER = 0.15


class CarlaWorld(World):
  """Synchronous CARLA world that supplies the same RGB/GPS/IMU contract as MetaDrive."""

  def __init__(self, status_q, *, host: str, port: int, town: str, scenario: str, dual_camera: bool,
               traffic_count: int, test_duration: float, test_run: bool):
    super().__init__(dual_camera)
    try:
      import carla
    except ImportError as exc:
      raise RuntimeError("CARLA Python API is unavailable; install carla==0.9.16 in the openpilot venv") from exc

    self.carla = carla
    self.status_q = status_q
    self.test_run = test_run
    self.test_duration = test_duration
    self.traffic_count = traffic_count
    self.scenario = scenario
    self._camera_frames: queue.Queue[Any] = queue.Queue(maxsize=2)
    self._collision_events: queue.Queue[Any] = queue.Queue()
    self._lane_events: queue.Queue[Any] = queue.Queue()
    self._actors: list[Any] = []
    self._sensors: list[Any] = []
    self._control = carla.VehicleControl()
    self._engaged_at: float | None = None
    self._last_control_at = time.monotonic()
    self._original_settings = None
    self._spectator = None
    self._route = []
    self._route_step_m = 2.0
    self._route_cursor_index = 0
    self._route_max_index = 0
    self._route_complete_sent = False
    self._city_stop_markers: list[int] = []
    self._city_stop_cursor = 0
    self._city_stop_until: float | None = None
    self._lane_center_errors_m: list[float] = []
    self._last_metrics_frame = -1
    self._scenario_info: dict[str, float | int | str] = {"scenario": scenario}
    self._initial_speed_applied = False
    self._initial_speed_mps = SCENARIO_ENTRY_SPEED_MPS.get(scenario, 0.0)
    self._carla_max_steer_deg = 1.0
    self._steering_wheel_limit_deg = 1.0
    self._telemetry_previous_yaw_deg: float | None = None
    self._telemetry_previous_frame = -1
    self._lead = None
    self._lead_base_speed_mps = 0.0
    self._lead_commanded_speed_mps = 0.0
    self._specialist = None
    self._specialist_prediction_1pm: float | None = None
    self._specialist_normalized_steer: float | None = None
    self._control_source = "openpilot"
    self._ground_truth_mode = os.environ.get("CARLA_GROUND_TRUTH_MODE", "none")
    if self._ground_truth_mode not in {"none", "lateral", "longitudinal", "both"}:
      raise RuntimeError("CARLA_GROUND_TRUTH_MODE must be none, lateral, longitudinal, or both")
    self._ground_truth_normalized_steer: float | None = None
    self._ground_truth_reference_curvature_1pm: float | None = None
    self._ground_truth_target_speed_mps: float | None = None
    self._ground_truth_lateral_error_m: float | None = None
    self._ground_truth_heading_error_deg: float | None = None
    self._ground_truth_speed_integral = 0.0
    self._ground_truth_previous_speed_error: float | None = None
    self._ground_truth_last_control_at: float | None = None
    specialist_path = os.environ.get("CARLA_SPECIALIST_MODEL")
    if specialist_path:
      from openpilot.tools.sim.carla_specialist.model import SpecialistModel
      self._specialist = SpecialistModel.load(specialist_path)
    self._specialist_mode = os.environ.get("CARLA_SPECIALIST_MODE", "shadow" if self._specialist is not None else "disabled")
    if self._specialist_mode not in {"disabled", "shadow", "assist"}:
      raise RuntimeError("CARLA_SPECIALIST_MODE must be disabled, shadow, or assist")
    if self._specialist_mode == "assist" and self._specialist is None:
      raise RuntimeError("CARLA_SPECIALIST_MODE=assist requires CARLA_SPECIALIST_MODEL")
    try:
      camera_pitch_deg = float(os.environ.get("CARLA_CAMERA_PITCH_DEG", str(CAMERA_MOUNT[3])))
    except ValueError as exc:
      raise RuntimeError("CARLA_CAMERA_PITCH_DEG must be numeric") from exc
    self._camera_mount = (CAMERA_MOUNT[0], CAMERA_MOUNT[1], CAMERA_MOUNT[2], camera_pitch_deg)
    capture_dir = os.environ.get("CARLA_CAPTURE_DIR")
    self._capture_dir = Path(capture_dir) if capture_dir else None
    self._next_capture_at: float | None = None
    self._capture_index = 0

    self.client = carla.Client(host, port)
    # Map switches are normally completed by the launcher before this bridge
    # starts. Keep a longer fallback timeout for direct bridge invocation.
    self.client.set_timeout(120.0)
    self.world = self.client.get_world()
    if self.world.get_map().name.rsplit("/", 1)[-1] != town:
      self.world = self.client.load_world(town)

    self._original_settings = self.world.get_settings()
    settings = self.world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    self.world.apply_settings(settings)
    self._spectator = self.world.get_spectator()
    self._spawn_ego()
    self._spawn_traffic()
    self.status_q.put(QueueMessage(QueueMessageType.START_STATUS, "started"))

  def _put_latest(self, target: queue.Queue[Any], value: Any):
    try:
      target.put_nowait(value)
    except queue.Full:
      try:
        target.get_nowait()
      except queue.Empty:
        pass
      target.put_nowait(value)

  def _spawn_ego(self):
    bp_lib = self.world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = self.world.get_map().get_spawn_points()
    if not spawn_points:
      raise RuntimeError("CARLA map has no vehicle spawn points")
    selected_transform = spawn_points[0]
    if self.scenario == "curve_60s":
      selected_transform = self._select_curve_route(spawn_points)
    elif self.scenario == "city_mixed":
      selected_transform = self._load_city_route_asset()
    elif self.scenario in {"straight_30", "straight_50", "straight_70", *LEAD_SCENARIOS}:
      selected_transform = self._select_straight_route(spawn_points)
    self.ego = self.world.try_spawn_actor(vehicle_bp, selected_transform)
    self._ego_spawn_transform = selected_transform
    if self.ego is None:
      for transform in spawn_points:
        self.ego = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.ego is not None:
          self._ego_spawn_transform = transform
          break
    if self.ego is None:
      raise RuntimeError("unable to spawn CARLA ego vehicle")
    if self.scenario == "city_mixed" and self._route:
      route_start = self._route[0].transform.location
      if self.ego.get_location().distance(route_start) > 5.0:
        raise RuntimeError("city route asset spawn did not align with ego; refusing to run an invalid route")
    self._actors.append(self.ego)
    front_wheel_limits = [wheel.max_steer_angle for wheel in self.ego.get_physics_control().wheels if wheel.max_steer_angle > 0]
    if front_wheel_limits:
      self._carla_max_steer_deg = max(front_wheel_limits)
    self._steering_wheel_limit_deg = self._carla_max_steer_deg * HONDA_CIVIC_2022_STEER_RATIO
    self._scenario_info.update({
      "camera_mount_x_m": self._camera_mount[0],
      "camera_mount_y_m": self._camera_mount[1],
      "camera_mount_z_m": self._camera_mount[2],
      "camera_pitch_deg": self._camera_mount[3],
      "camera_hfov_deg": round(CAMERA_HORIZONTAL_FOV_DEG, 4),
      "camera_focal_length_px": CAMERA_FOCAL_LENGTH_PX,
      "carla_max_front_wheel_deg": self._carla_max_steer_deg,
      "steering_wheel_limit_deg": round(self._steering_wheel_limit_deg, 4),
      "carla_specialist_mode": self._specialist_mode,
    })
    if self._initial_speed_mps:
      # Apply the controlled entry speed only after openpilot is active. The
      # bridge performs startup ticks before the controller exists; moving now
      # would drive an unsteered car into the curve.
      self._scenario_info["initial_speed_mps"] = self._initial_speed_mps

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(W))
    camera_bp.set_attribute("image_size_y", str(H))
    camera_bp.set_attribute("fov", f"{CAMERA_HORIZONTAL_FOV_DEG:.6f}")
    camera_bp.set_attribute("sensor_tick", "0.05")
    camera = self.world.spawn_actor(
      camera_bp,
      self.carla.Transform(
        self.carla.Location(x=self._camera_mount[0], y=self._camera_mount[1], z=self._camera_mount[2]),
        self.carla.Rotation(pitch=self._camera_mount[3]),
      ),
      attach_to=self.ego,
    )
    camera.listen(self._put_camera_frame)
    self._sensors.append(camera)

    collision = self.world.spawn_actor(bp_lib.find("sensor.other.collision"), self.carla.Transform(), attach_to=self.ego)
    collision.listen(self._collision_events.put)
    self._sensors.append(collision)
    lane = self.world.spawn_actor(bp_lib.find("sensor.other.lane_invasion"), self.carla.Transform(), attach_to=self.ego)
    lane.listen(self._lane_events.put)
    self._sensors.append(lane)

  @staticmethod
  def _yaw_delta_deg(before: float, after: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0

  @staticmethod
  def _unit_xy(vector):
    magnitude = math.hypot(vector.x, vector.y)
    if magnitude <= 1e-6:
      raise RuntimeError("CARLA actor has an invalid horizontal forward vector")
    return vector.x / magnitude, vector.y / magnitude

  def _trace_route(self, start, distance_m: float = 320.0):
    route = [start]
    steps = int(distance_m / self._route_step_m)
    for _ in range(steps):
      choices = route[-1].next(self._route_step_m)
      if not choices:
        break
      previous_yaw = route[-1].transform.rotation.yaw
      route.append(min(choices, key=lambda item: abs(self._yaw_delta_deg(previous_yaw, item.transform.rotation.yaw))))
    return route

  def _select_curve_route(self, spawn_points):
    carla_map = self.world.get_map()
    best_route = []
    best_score = -1.0
    best_spawn_index = 0
    # Waypoint.next() is an RPC. Sample the map deterministically instead of
    # walking every spawn point, keeping scenario startup bounded on WSL.
    stride = max(1, math.ceil(len(spawn_points) / 20))
    for index in range(0, len(spawn_points), stride):
      transform = spawn_points[index]
      waypoint = carla_map.get_waypoint(transform.location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
      if waypoint is None:
        continue
      route = self._trace_route(waypoint)
      if len(route) < 80:
        continue
      curvature = sum(abs(self._yaw_delta_deg(a.transform.rotation.yaw, b.transform.rotation.yaw)) for a, b in zip(route, route[1:]))
      score = curvature + len(route) * 0.01
      if score > best_score:
        best_route, best_score, best_spawn_index = route, score, index
    if not best_route:
      waypoint = carla_map.get_waypoint(spawn_points[0].location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
      best_route = self._trace_route(waypoint) if waypoint is not None else []
    self._route = best_route
    self._scenario_info.update({
      "route_length_m": round(max(0, len(best_route) - 1) * self._route_step_m, 1),
      "route_curve_deg": round(best_score if best_score >= 0 else 0.0, 2),
      "route_spawn_index": best_spawn_index,
    })
    return best_route[0].transform if best_route else spawn_points[0]

  def _select_straight_route(self, spawn_points):
    carla_map = self.world.get_map()
    best_route = []
    best_score = math.inf
    best_spawn_index = 0
    stride = max(1, math.ceil(len(spawn_points) / 20))
    for index in range(0, len(spawn_points), stride):
      waypoint = carla_map.get_waypoint(spawn_points[index].location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
      if waypoint is None:
        continue
      route = self._trace_route(waypoint)
      if len(route) < 80:
        continue
      score = sum(abs(self._yaw_delta_deg(a.transform.rotation.yaw, b.transform.rotation.yaw)) for a, b in zip(route, route[1:]))
      if score < best_score:
        best_route, best_score, best_spawn_index = route, score, index
    if not best_route:
      return spawn_points[0]
    self._route = best_route
    self._scenario_info.update({
      "route_length_m": round(max(0, len(best_route) - 1) * self._route_step_m, 1),
      "route_curve_deg": round(best_score, 2),
      "route_spawn_index": best_spawn_index,
    })
    return best_route[0].transform

  def _load_city_route_asset(self):
    """Load a precomputed route without any startup waypoint-RPC traversal."""
    asset_path = os.environ.get("CARLA_CITY_ROUTE_ASSET")
    if not asset_path:
      raise RuntimeError("city_mixed requires CARLA_CITY_ROUTE_ASSET; generate it before starting the synchronous bridge")
    asset = json.loads(Path(asset_path).read_text(encoding="utf-8"))
    if asset.get("town") != self.world.get_map().name.rsplit("/", 1)[-1]:
      raise RuntimeError(f"city route asset town mismatch: {asset.get('town')} != {self.world.get_map().name}")
    points = asset.get("points", [])
    if len(points) < 80:
      raise RuntimeError("city route asset has too few points")
    self._route = [SimpleNamespace(transform=self.carla.Transform(
      self.carla.Location(x=point[0], y=point[1], z=point[2]),
      self.carla.Rotation(pitch=point[3], yaw=point[4], roll=point[5]),
    )) for point in points]
    self._city_stop_markers = [int((len(self._route) - 1) * fraction) for fraction in (0.25, 0.55, 0.82)]
    self._scenario_info.update({
      "route_length_m": round((len(self._route) - 1) * self._route_step_m, 1),
      "city_turn_plan": asset.get("turn_plan", list(CITY_TURN_PLAN)),
      "city_turns_realized": asset.get("turns_realized", []),
      "city_route_asset": asset_path,
      "city_stop_markers_m": [round(marker * self._route_step_m, 1) for marker in self._city_stop_markers],
      "city_stop_hold_seconds": 3.0,
    })
    start = self._route[0].transform.location
    # Use CARLA's own nearest spawn transform, not a centreline waypoint
    # transform. The latter can be rejected even when its XY position matches.
    return min(self.world.get_map().get_spawn_points(), key=lambda item: item.location.distance(start))

  def _trace_city_route(self, start, distance_m: float = 500.0):
    """Trace a deterministic urban route with explicit intersection choices.

    A U-turn is represented by three successive left turns around a city block;
    this is repeatable on CARLA town maps where a literal mid-lane U-turn is
    not a legal waypoint transition.
    """
    route, plan_index, completed = [start], 0, []
    expanded_plan = ("straight", "left", "straight", "right", "straight", "left", "left", "left", "straight")
    for _ in range(int(distance_m / self._route_step_m)):
      choices = route[-1].next(self._route_step_m)
      if not choices:
        break
      previous_yaw = route[-1].transform.rotation.yaw
      deltas = [self._yaw_delta_deg(previous_yaw, item.transform.rotation.yaw) for item in choices]
      directive = expanded_plan[min(plan_index, len(expanded_plan) - 1)]
      if len(choices) > 1 and max(deltas) - min(deltas) >= 15.0:
        if directive == "left":
          chosen = choices[deltas.index(min(deltas))]
        elif directive == "right":
          chosen = choices[deltas.index(max(deltas))]
        else:
          chosen = choices[min(range(len(choices)), key=lambda index: abs(deltas[index]))]
        if plan_index < len(expanded_plan):
          plan_index += 1
          if plan_index == 2:
            completed.append("left_turn")
          elif plan_index == 4:
            completed.append("right_turn")
          elif plan_index == 8:
            completed.append("u_turn_via_three_left_turns")
      else:
        chosen = choices[min(range(len(choices)), key=lambda index: abs(deltas[index]))]
      route.append(chosen)
    return route, completed

  def _select_city_route(self, spawn_points):
    carla_map = self.world.get_map()
    best_route, best_completed, best_spawn_index = [], [], 0
    # Search a bounded, deterministic subset so startup remains practical.
    stride = max(1, math.ceil(len(spawn_points) / 24))
    for index in range(0, len(spawn_points), stride):
      waypoint = carla_map.get_waypoint(spawn_points[index].location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
      if waypoint is None:
        continue
      route, completed = self._trace_city_route(waypoint)
      score = len(completed) * 10000 + len(route)
      best_score = len(best_completed) * 10000 + len(best_route)
      if len(route) >= 100 and score > best_score:
        best_route, best_completed, best_spawn_index = route, completed, index
    if not best_route:
      waypoint = carla_map.get_waypoint(spawn_points[0].location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
      best_route, best_completed = self._trace_city_route(waypoint) if waypoint else ([], [])
    self._route = best_route
    self._city_stop_markers = [int((len(best_route) - 1) * fraction) for fraction in (0.25, 0.55, 0.82)]
    self._scenario_info.update({
      "route_length_m": round(max(0, len(best_route) - 1) * self._route_step_m, 1),
      "route_spawn_index": best_spawn_index,
      "city_turn_plan": list(CITY_TURN_PLAN),
      "city_turns_realized": best_completed,
      "city_stop_markers_m": [round(marker * self._route_step_m, 1) for marker in self._city_stop_markers],
      "city_stop_hold_seconds": 3.0,
    })
    return best_route[0].transform if best_route else spawn_points[0]

  def _spawn_traffic(self):
    if self.scenario in LEAD_SCENARIOS:
      self._spawn_lead()
    if self.traffic_count <= 0:
      return
    blueprints = self.world.get_blueprint_library().filter("vehicle.*")
    ego_location = self.ego.get_location()
    # Do not materialize traffic on the ego's entry road. A collision during
    # the first few simulator ticks is a spawn artefact, not traffic behavior.
    spawn_points = [point for point in self.world.get_map().get_spawn_points()
                    if point.location.distance(ego_location) >= 40.0]
    traffic_manager = self.client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    for transform in spawn_points[:self.traffic_count]:
      actor = self.world.try_spawn_actor(blueprints[len(self._actors) % len(blueprints)], transform)
      if actor is not None:
        actor.set_autopilot(True, traffic_manager.get_port())
        self._actors.append(actor)

  def _spawn_lead(self):
    # This scenario has already selected a straight ego segment. Place the
    # lead directly along its forward vector instead of following a Town04
    # waypoint branch, which can jump to an overlapping carriageway.
    # Actor transforms are not reliable until the first synchronous tick; use
    # the selected spawn transform while constructing the pre-tick lead.
    ego_transform = self._ego_spawn_transform
    forward = ego_transform.get_forward_vector()
    forward_x, forward_y = self._unit_xy(forward)
    transform = self.carla.Transform(
      self.carla.Location(x=ego_transform.location.x + 35.0 * forward_x, y=ego_transform.location.y + 35.0 * forward_y,
                          z=ego_transform.location.z),
      self.carla.Rotation(pitch=ego_transform.rotation.pitch, yaw=ego_transform.rotation.yaw, roll=ego_transform.rotation.roll),
    )
    bp = self.world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    self._lead = self.world.try_spawn_actor(bp, transform)
    if self._lead is None:
      raise RuntimeError("unable to spawn deterministic lead vehicle")
    self._actors.append(self._lead)
    self._lead.set_simulate_physics(False)
    self._scenario_info["lead_spawn_requested_distance_m"] = 35.0
    self._lead_base_speed_mps = 30.0 / 3.6 if self.scenario == "lead_follow_30" else 8.0
    self._scenario_info.update({
      "lead_initial_distance_m": 35.0,
      "lead_base_speed_mps": self._lead_base_speed_mps,
      "lead_control": "kinematic CARLA evaluator actor (never supplied to openpilot)",
    })

  def _put_camera_frame(self, image):
    raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))
    # CARLA returns BGRA; openpilot's camerad conversion expects RGB.
    self._put_latest(self._camera_frames, raw[:, :, :3][:, :, ::-1].copy())

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    normalized_steer = float(np.clip(steer_angle / self._steering_wheel_limit_deg, -1.0, 1.0))
    throttle, brake = float(np.clip(throttle_out, 0.0, 1.0)), float(np.clip(brake_out, 0.0, 1.0))
    self._control_source = "openpilot"
    if self._specialist_mode == "assist" and self._engaged_at is not None and self._specialist_prediction_1pm is not None:
      # Curvature labels are in openpilot's lateral convention, opposite
      # CARLA's steer/yaw sign. Low-pass and bound output to make this a
      # controlled simulator experiment, not an unconstrained override.
      requested = float(np.clip(-self._specialist_prediction_1pm / SPECIALIST_CURVATURE_PER_NORMALIZED_STEER,
                                -SPECIALIST_MAX_NORMALIZED_STEER, SPECIALIST_MAX_NORMALIZED_STEER))
      previous = self._specialist_normalized_steer if self._specialist_normalized_steer is not None else requested
      self._specialist_normalized_steer = 0.8 * previous + 0.2 * requested
      normalized_steer = self._specialist_normalized_steer
      self._control_source = "sim_specialist_assist"
    if self._engaged_at is not None and self._ground_truth_mode in {"lateral", "both"}:
      reference = self._route_reference_curvature()
      if reference is not None:
        self._ground_truth_reference_curvature_1pm = reference
        requested = float(np.clip(-reference / SPECIALIST_CURVATURE_PER_NORMALIZED_STEER,
                                  -GROUND_TRUTH_MAX_NORMALIZED_STEER, GROUND_TRUTH_MAX_NORMALIZED_STEER))
        lateral_error, heading_error = self._ground_truth_tracking_error()
        self._ground_truth_lateral_error_m = lateral_error
        self._ground_truth_heading_error_deg = heading_error
        # Feed-forward follows the map curvature. These two feedback terms
        # bring the vehicle back after a road-edge disturbance instead of
        # continuing the same-radius turn into the outside barrier.
        requested += float(np.clip(-0.080 * lateral_error, -0.080, 0.080))
        requested += float(np.clip(0.012 * heading_error, -0.080, 0.080))
        requested = float(np.clip(requested, -GROUND_TRUTH_MAX_NORMALIZED_STEER, GROUND_TRUTH_MAX_NORMALIZED_STEER))
        previous = self._ground_truth_normalized_steer if self._ground_truth_normalized_steer is not None else requested
        # The map curvature is sampled at the simulator tick rate and can jump
        # at waypoint joins. Limit its slew before the CARLA wheel actuator.
        requested = float(np.clip(requested, previous - 0.004, previous + 0.004))
        self._ground_truth_normalized_steer = 0.85 * previous + 0.15 * requested
        normalized_steer = self._ground_truth_normalized_steer
        self._control_source = "carla_ground_truth_lateral"
    if self._engaged_at is not None and self._ground_truth_mode in {"longitudinal", "both"}:
      velocity = self.ego.get_velocity()
      speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
      reference = self._route_reference_curvature()
      # A lateral-acceleration speed cap provides a repeatable longitudinal
      # target without looking at the learned driving model.  It is map truth
      # for this simulator experiment only.
      now = time.monotonic()
      curve_cap = math.sqrt(0.35 / max(abs(reference or 0.0), 1e-3))
      target = min(self._initial_speed_mps, curve_cap)
      if self.scenario == "city_mixed" and self._city_stop_cursor < len(self._city_stop_markers):
        marker = self._city_stop_markers[self._city_stop_cursor]
        if self._route_max_index >= marker:
          if self._city_stop_until is None:
            self._city_stop_until = now + 3.0
          if now < self._city_stop_until:
            target = 0.0
          else:
            self._city_stop_cursor += 1
            self._city_stop_until = None
            self._scenario_info["city_stops_completed"] = self._city_stop_cursor
      self._ground_truth_target_speed_mps = target
      error = target - speed
      dt = min(0.1, max(0.01, now - self._ground_truth_last_control_at)) if self._ground_truth_last_control_at else 0.05
      self._ground_truth_last_control_at = now
      self._ground_truth_speed_integral = float(np.clip(self._ground_truth_speed_integral + error * dt, -4.0, 4.0))
      derivative = 0.0 if self._ground_truth_previous_speed_error is None else (error - self._ground_truth_previous_speed_error) / dt
      self._ground_truth_previous_speed_error = error
      effort = 0.18 * error + 0.035 * self._ground_truth_speed_integral + 0.012 * derivative
      throttle = float(np.clip(effort, 0.0, 0.50))
      brake = float(np.clip(-effort, 0.0, 0.45))
      self._control_source = ("carla_ground_truth_both" if self._ground_truth_mode == "both"
                              else "carla_ground_truth_longitudinal")
    self._control = self.carla.VehicleControl(
      throttle=throttle,
      brake=brake,
      # CARLA consumes normalized front-wheel angle. Convert the Honda
      # steering-wheel command through its steer ratio and CARLA's measured
      # wheel limit instead of the old arbitrary 70-degree divisor.
      steer=normalized_steer,
    )
    self._last_control_at = time.monotonic()
    self.ego.apply_control(self._control)

  def tick(self):
    self._update_lead_velocity()
    self.world.tick()
    self._follow_ego_from_driver_seat()
    self._emit_physics_telemetry()

  def _update_lead_velocity(self):
    if self._lead is None:
      return
    # The bridge performs CARLA warm-up ticks before openpilot is engaged.
    # Do not advance an evaluator lead during that period: synchronous CARLA
    # time can progress faster than wall time and otherwise invalidates the
    # intended 35 m initial gap.
    if self._engaged_at is None:
      self._lead_commanded_speed_mps = 0.0
      return
    speed_mps = self._lead_base_speed_mps
    if self.scenario == "lead_stop_restart" and self._engaged_at is not None:
      elapsed = time.monotonic() - self._engaged_at
      if 8.0 <= elapsed < 12.0:
        speed_mps = 0.0
    self._lead_commanded_speed_mps = speed_mps
    forward = self._lead.get_transform().get_forward_vector()
    forward_x, forward_y = self._unit_xy(forward)
    transform = self._lead.get_transform()
    transform.location.x += speed_mps * forward_x * 0.05
    transform.location.y += speed_mps * forward_y * 0.05
    self._lead.set_transform(transform)

  def _emit_physics_telemetry(self):
    """Publish physical CARLA response, separate from simulated CAN feedback."""
    snapshot = self.world.get_snapshot()
    if snapshot.frame == self._telemetry_previous_frame:
      return
    self._telemetry_previous_frame = snapshot.frame
    transform = self.ego.get_transform()
    velocity = self.ego.get_velocity()
    speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    yaw_rate_radps = None
    if self._telemetry_previous_yaw_deg is not None:
      delta_seconds = max(float(snapshot.timestamp.delta_seconds), 1e-3)
      yaw_rate_radps = math.radians(self._yaw_delta_deg(self._telemetry_previous_yaw_deg, transform.rotation.yaw)) / delta_seconds
    self._telemetry_previous_yaw_deg = transform.rotation.yaw
    curvature_1pm = yaw_rate_radps / speed_mps if yaw_rate_radps is not None and speed_mps > 0.1 else None
    lead_distance_m = lead_longitudinal_distance_m = lead_relative_speed_mps = lead_ttc_s = None
    if self._lead is not None:
      ego_location = self.ego.get_location()
      lead_location = self._lead.get_location()
      forward = transform.get_forward_vector()
      forward_x, forward_y = self._unit_xy(forward)
      delta_x, delta_y = lead_location.x - ego_location.x, lead_location.y - ego_location.y
      lead_distance_m = math.hypot(delta_x, delta_y)
      lead_longitudinal_distance_m = delta_x * forward_x + delta_y * forward_y
      lead_forward = self._lead.get_transform().get_forward_vector()
      lead_forward_x, lead_forward_y = self._unit_xy(lead_forward)
      lead_speed_along_ego = self._lead_commanded_speed_mps * (lead_forward_x * forward_x + lead_forward_y * forward_y)
      ego_speed_along_ego = velocity.x * forward_x + velocity.y * forward_y
      lead_relative_speed_mps = lead_speed_along_ego - ego_speed_along_ego
      if lead_longitudinal_distance_m > 0.0 and lead_relative_speed_mps < -0.1:
        lead_ttc_s = lead_longitudinal_distance_m / -lead_relative_speed_mps
    self.status_q.put(QueueMessage(QueueMessageType.TELEMETRY, {
      "carla_normalized_steer": float(self._control.steer),
      "carla_speed_mps": speed_mps,
      "carla_actual_yaw_rate_radps": yaw_rate_radps,
      "carla_actual_curvature_1pm": curvature_1pm,
      "carla_route_reference_curvature_1pm": self._route_reference_curvature(),
      "carla_lead_distance_m": lead_distance_m,
      "carla_lead_longitudinal_distance_m": lead_longitudinal_distance_m,
      "carla_lead_relative_speed_mps": lead_relative_speed_mps,
      "carla_lead_ttc_s": lead_ttc_s,
      "scenario_target_speed_mps": self._ground_truth_target_speed_mps if self._ground_truth_target_speed_mps is not None else self._initial_speed_mps or None,
      "carla_specialist_curvature_1pm": self._specialist_prediction_1pm,
      "carla_specialist_normalized_steer": self._specialist_normalized_steer,
      "carla_ground_truth_reference_curvature_1pm": self._ground_truth_reference_curvature_1pm,
      "carla_ground_truth_target_speed_mps": self._ground_truth_target_speed_mps,
      "carla_ground_truth_lateral_error_m": self._ground_truth_lateral_error_m,
      "carla_ground_truth_heading_error_deg": self._ground_truth_heading_error_deg,
      "carla_control_source": self._control_source,
    }))

  def _ground_truth_tracking_error(self) -> tuple[float, float]:
    """Return CARLA-right-positive centre error and desired-minus-current yaw."""
    if not self._route:
      return 0.0, 0.0
    location = self.ego.get_location()
    low = max(0, self._route_cursor_index - 8)
    high = min(len(self._route), self._route_cursor_index + 30)
    nearest = min(range(low, high), key=lambda index: location.distance(self._route[index].transform.location))
    target = self._route[nearest].transform
    dx, dy = location.x - target.location.x, location.y - target.location.y
    right = target.get_right_vector()
    lateral_error = dx * right.x + dy * right.y
    heading_error = self._yaw_delta_deg(self.ego.get_transform().rotation.yaw, target.rotation.yaw)
    return float(lateral_error), float(heading_error)

  def _route_reference_curvature(self):
    """Signed centre-line curvature ahead of the ego for evaluator diagnostics.

    This is ground truth from CARLA's road graph, never an input to openpilot
    or the vehicle controls. Keeping it telemetry-only lets the harness tell a
    perception failure (model target differs from the road) from a controller
    or vehicle-response failure.
    """
    if len(self._route) < 3:
      return None
    location = self.ego.get_location()
    low = max(0, self._route_cursor_index - 5)
    high = min(len(self._route), self._route_cursor_index + 25)
    nearest = min(range(low, high), key=lambda index: location.distance(self._route[index].transform.location))
    before = max(0, nearest - 2)
    after = min(len(self._route) - 1, nearest + 12)
    distance_m = (after - before) * self._route_step_m
    if distance_m <= 0.0:
      return None
    delta_yaw_deg = self._yaw_delta_deg(self._route[before].transform.rotation.yaw, self._route[after].transform.rotation.yaw)
    # CARLA's +Y/right-handed road heading and openpilot's vehicle frame use
    # opposite lateral signs. Normalise the evaluator reference to openpilot's
    # curvature convention before the harness compares it to modelV2.
    return -math.radians(delta_yaw_deg) / distance_m

  def _follow_ego_from_driver_seat(self):
    """Move only CARLA's display spectator; never alter the openpilot sensor."""
    if self._spectator is None:
      return
    vehicle = self.ego.get_transform()
    forward = vehicle.get_forward_vector()
    right = vehicle.get_right_vector()
    # Approximate left-hand-drive seat in the Tesla Model 3 local frame.
    # The openpilot RGB camera remains at its separately calibrated mount.
    location = self.carla.Location(
      x=vehicle.location.x + 0.45 * forward.x - 0.35 * right.x,
      y=vehicle.location.y + 0.45 * forward.y - 0.35 * right.y,
      z=vehicle.location.z + 1.30,
    )
    rotation = self.carla.Rotation(pitch=0.0, yaw=vehicle.rotation.yaw, roll=0.0)
    self._spectator.set_transform(self.carla.Transform(location, rotation))

  def read_state(self):
    self._record_route_metrics()
    if not self._collision_events.empty():
      self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"collision": True, **self._scenario_metrics()}))
      if self.test_run:
        self.exit_event.set()
    if not self._lane_events.empty():
      event_count = 0
      while not self._lane_events.empty():
        self._lane_events.get_nowait()
        event_count += 1
      self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"lane_invasion_count": event_count}))
    if self.test_run and self._engaged_at is not None and time.monotonic() - self._engaged_at >= self.test_duration:
      self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {"timeout": True, **self._scenario_metrics()}))
      self.exit_event.set()

  def _record_route_metrics(self):
    if self.scenario not in {"curve_60s", "city_mixed"}:
      return
    frame = self.world.get_snapshot().frame
    if frame == self._last_metrics_frame:
      return
    self._last_metrics_frame = frame
    location = self.ego.get_location()
    waypoint = self.world.get_map().get_waypoint(location, project_to_road=True, lane_type=self.carla.LaneType.Driving)
    if waypoint is not None:
      self._lane_center_errors_m.append(float(location.distance(waypoint.transform.location)))
    if self._route:
      # Some Town04 roads overlap in world space. Constrain projection to the
      # local route neighbourhood so a stationary vehicle cannot jump ahead to
      # a later crossing of the same road.
      low = max(0, self._route_cursor_index - 5)
      high = min(len(self._route), self._route_cursor_index + 25)
      nearest = min(range(low, high), key=lambda index: location.distance(self._route[index].transform.location))
      self._route_cursor_index = nearest
      self._route_max_index = max(self._route_max_index, nearest)
      route_length = (len(self._route) - 1) * self._route_step_m
      progress = self._route_max_index * self._route_step_m
      # curve_60s has a finite, map-derived route. Continuing to apply a
      # controller after its final waypoint turns a completed route into an
      # off-route wall collision, which is not a control-quality measurement.
      if self.test_run and not self._route_complete_sent and route_length > 0.0 and progress >= route_length - 4.0:
        self._route_complete_sent = True
        self.ego.apply_control(self.carla.VehicleControl(brake=1.0))
        self.status_q.put(QueueMessage(QueueMessageType.TERMINATION_INFO, {
          "route_complete": True, **self._scenario_metrics(),
        }))
        self.exit_event.set()

  def _scenario_metrics(self):
    route_length = max(0.0, (len(self._route) - 1) * self._route_step_m)
    progress = min(route_length, self._route_max_index * self._route_step_m)
    errors = self._lane_center_errors_m
    return {
      **self._scenario_info,
      "route_progress_m": round(progress, 2),
      "route_progress_pct": round(100.0 * progress / route_length, 2) if route_length else None,
      "mean_lane_center_error_m": round(float(np.mean(errors)), 4) if errors else None,
      "max_lane_center_error_m": round(max(errors), 4) if errors else None,
    }

  def read_sensors(self, state: SimulatorState):
    transform = self.ego.get_transform()
    velocity = self.ego.get_velocity()
    state.velocity = vec3(float(velocity.x), float(velocity.y), float(velocity.z))
    state.bearing = float(transform.rotation.yaw)
    state.steering_angle = self._control.steer * self._steering_wheel_limit_deg
    state.gps.from_xy((transform.location.x, transform.location.y))
    # CARLA reports IMU-like dynamics in Unreal world coordinates, whereas
    # openpilot consumes a calibrated device frame. Until a rigid C3 mounting
    # transform is part of this evaluator, use the same zero-motion virtual
    # IMU contract as the proven MetaDrive bridge and keep GPS/CAN velocity
    # authoritative. Raw world-frame acceleration makes locationd infer
    # impossible device motion.
    state.imu.accelerometer = vec3(0.0, 0.0, 0.0)
    state.imu.gyroscope = vec3(0.0, 0.0, 0.0)
    state.imu.bearing = 0.0
    state.valid = True
    if state.is_engaged and self._engaged_at is None:
      self._engaged_at = time.monotonic()
    if state.is_engaged and not self._initial_speed_applied and self._initial_speed_mps:
      forward = transform.get_forward_vector()
      self.ego.set_target_velocity(self.carla.Vector3D(
        x=self._initial_speed_mps * forward.x, y=self._initial_speed_mps * forward.y, z=0.0,
      ))
      self._initial_speed_applied = True

  def read_cameras(self):
    try:
      image = self._camera_frames.get(timeout=2.0)
    except queue.Empty as exc:
      raise RuntimeError("CARLA camera did not produce a frame within 2 seconds") from exc
    self.road_image[...] = image
    if self._specialist is not None:
      # Shadow-only: this prediction is never substituted for modelV2 or
      # passed to openpilot's planner/controller.
      self._specialist_prediction_1pm = self._specialist.predict(image)
    self._capture_camera_frame(image)
    self.image_lock.release()

  def _capture_camera_frame(self, image: np.ndarray):
    """Persist sparse evaluator-only camera frames when the harness requests it."""
    if self._capture_dir is None or self._engaged_at is None:
      return
    elapsed = time.monotonic() - self._engaged_at
    if self._next_capture_at is not None and elapsed < self._next_capture_at:
      return
    self._capture_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(image).save(self._capture_dir / f"active-{self._capture_index:02d}-{elapsed:05.1f}s.png")
    self._capture_index += 1
    self._next_capture_at = elapsed + 5.0

  def reset(self):
    self.ego.apply_control(self.carla.VehicleControl(brake=1.0))

  def close(self, reason: str):
    self.status_q.put(QueueMessage(QueueMessageType.CLOSE_STATUS, reason))
    self.exit_event.set()
    for sensor in self._sensors:
      try:
        sensor.stop()
        sensor.destroy()
      except RuntimeError:
        pass
    for actor in self._actors:
      try:
        actor.destroy()
      except RuntimeError:
        pass
    if self._original_settings is not None:
      try:
        self.world.apply_settings(self._original_settings)
      except RuntimeError:
        pass
