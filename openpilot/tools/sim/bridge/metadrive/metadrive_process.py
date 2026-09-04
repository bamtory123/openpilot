import math
import os
import time
import json
import numpy as np

from collections import namedtuple
from panda3d.core import Point2, Point3, Vec3
from PIL import Image
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from openpilot.common.realtime import Ratekeeper

from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H
from openpilot.tools.sim.bridge.metadrive.metadrive_common import apply_camera_color_affine
from openpilot.tools.sim.bridge.metadrive.specialist_replay import SpecialistReplay

C3_POSITION = Vec3(0.0, 0, 1.22)
C3_HPR = Vec3(0, 0,0)


metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle", "ground_truth"])


def world_signed_angle(forward: np.ndarray, target: np.ndarray) -> float:
  """Signed angle from a vehicle world-frame direction to a world-frame target."""
  return math.atan2(float(forward[0] * target[1] - forward[1] * target[0]), float(np.dot(forward, target)))


def apply_metadrive_patches(arrive_dest_done=True):
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
      sensor = cls(*args, self, cuda=True)
    else:
      sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  if not arrive_dest_done:
    MetaDriveEnv._is_arrive_destination = arrive_destination_patch

def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
                      controls_recv: Connection, simulation_state_send: Connection, vehicle_state_send: Connection,
                      exit_event, op_engaged, test_duration, test_run):
  arrive_dest_done = config.pop("arrive_dest_done", True)
  simlab_config = config.pop("simlab", {})
  environment_config = simlab_config.get("environment", {})
  simulator_control = simlab_config.get("simulator_control")
  specialist_dataset = simlab_config.get("specialist_dataset")
  specialist_replay_config = simlab_config.get("specialist_replay")
  actuation_config = simlab_config.get("actuation", {})
  reference_lane_index = int(environment_config.get("reference_lane_index", 0))
  seed = environment_config.get("seed")
  apply_metadrive_patches(arrive_dest_done)

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    assert wide_camera_array is not None
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MetaDriveEnv(config)
  specialist_replay = SpecialistReplay(specialist_replay_config["artifact_path"]) if specialist_replay_config is not None else None
  camera_fov_deg = float(environment_config.get("camera_fov_deg", 40))
  camera_gamma = float(environment_config.get("camera_gamma", 1.0))
  camera_color_affine = environment_config.get("camera_color_affine")
  camera_position = Vec3(*map(float, environment_config.get("camera_position_m", C3_POSITION)))
  camera_hpr = Vec3(*map(float, environment_config.get("camera_hpr_deg", C3_HPR)))
  debug_camera_path = os.environ.get("SIMLAB_CAMERA_DEBUG_PATH")
  debug_camera_metadata_path = os.environ.get("SIMLAB_CAMERA_DEBUG_METADATA_PATH") or (f"{debug_camera_path}.json" if debug_camera_path else None)
  debug_camera_after_frame = int(os.environ.get("SIMLAB_CAMERA_DEBUG_AFTER_FRAME", "0"))
  debug_capture_frames = [int(frame) for frame in os.environ.get("SIMLAB_CAMERA_DEBUG_CAPTURE_FRAMES", "").split(",") if frame]
  debug_capture_dir = os.environ.get("SIMLAB_CAMERA_DEBUG_CAPTURE_DIR")
  if debug_capture_frames and not debug_capture_dir:
    raise ValueError("SIMLAB_CAMERA_DEBUG_CAPTURE_DIR is required with SIMLAB_CAMERA_DEBUG_CAPTURE_FRAMES")
  debug_capture_index = 0
  debug_camera_captured = False
  controller_target_curvature = 0.0
  controller_lookahead_heading_error = 0.0
  previous_velocity_heading = None
  previous_simulation_time_s = None
  normalized_steer = 0.0
  configured_steer_ratio = float(actuation_config.get("steer_ratio", 8.0))

  def get_current_lane_info(vehicle):
    _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
    lane_idx = lane_info[2] if lane_info is not None else None
    return lane_idx, on_lane

  specialist_prediction = None
  lead_vehicle = lead_visual_proxy = None

  def reference_lane_telemetry(vehicle, simulation_frame, simulation_time_s, acceleration_mps2):
    nonlocal previous_velocity_heading, previous_simulation_time_s
    vehicle_width = float(vehicle.config.get("width") or 2.0)
    velocity_xy = np.asarray(vehicle.velocity[:2])
    speed_mps = float(np.linalg.norm(vehicle.velocity))
    yaw_rate = 0.0
    if speed_mps > 0.1:
      velocity_heading = math.atan2(float(velocity_xy[1]), float(velocity_xy[0]))
    else:
      velocity_heading = None
    if velocity_heading is not None and previous_velocity_heading is not None and previous_simulation_time_s is not None:
      dt = simulation_time_s - previous_simulation_time_s
      if dt > 0:
        yaw_rate = ((velocity_heading - previous_velocity_heading + math.pi) % (2 * math.pi) - math.pi) / dt
    if velocity_heading is not None:
      previous_velocity_heading = velocity_heading
      previous_simulation_time_s = simulation_time_s
    lanes = list(getattr(vehicle.navigation, "current_ref_lanes", []) or [])
    lane = lanes[reference_lane_index] if 0 <= reference_lane_index < len(lanes) else None
    lane_idx, on_lane = get_current_lane_info(vehicle)
    traffic_manager = getattr(env.engine, "traffic_manager", None)
    traffic_vehicle_count = int(traffic_manager.get_vehicle_num()) if traffic_manager is not None else 0
    active_traffic = list(getattr(traffic_manager, "traffic_vehicles", []) or []) if traffic_manager is not None else []
    if lead_vehicle is not None:
      traffic_vehicle_count += 1
      active_traffic.append(lead_vehicle)
    traffic_distances = [float(np.linalg.norm(np.asarray(other.position[:2]) - np.asarray(vehicle.position[:2]))) for other in active_traffic]
    nearest_traffic = active_traffic[int(np.argmin(traffic_distances))] if traffic_distances else None
    nearest_distance = min(traffic_distances) if traffic_distances else None
    traffic_closing_speed = traffic_ttc = None
    if nearest_traffic is not None and nearest_distance > 0:
      relative_position = np.asarray(nearest_traffic.position[:2]) - np.asarray(vehicle.position[:2])
      traffic_closing_speed = float(np.dot(np.asarray(vehicle.velocity[:2]) - np.asarray(nearest_traffic.velocity[:2]), relative_position / np.linalg.norm(relative_position)))
      traffic_ttc = float(nearest_distance / traffic_closing_speed) if traffic_closing_speed > 0 else None
    result = {
      "type": "vehicle_telemetry", "simulation_frame": simulation_frame, "simulation_time_s": simulation_time_s,
      "position_x_m": float(vehicle.position[0]), "position_y_m": float(vehicle.position[1]),
      "speed_mps": speed_mps, "acceleration_mps2": acceleration_mps2,
      "simulator_control_mode": "specialist_replay" if specialist_replay is not None else (simulator_control["mode"] if simulator_control is not None else "openpilot"),
      "specialist_replay_normalized_steer": specialist_prediction,
      "simulator_normalized_steer": float(normalized_steer),
      "configured_steer_ratio": configured_steer_ratio if simulator_control is None and specialist_replay is None else None,
      "applied_steering_angle_deg": float(vehicle.steering * vehicle.MAX_STEERING),
      "actual_yaw_rate_rad_s": yaw_rate,
      "actual_curvature_1pm": yaw_rate / speed_mps if speed_mps > 0.1 else 0.0,
      "controller_target_curvature_1pm": controller_target_curvature,
      "controller_lookahead_heading_error_rad": controller_lookahead_heading_error,
      "reference_tangent_world_x": None, "reference_tangent_world_y": None,
      "vehicle_velocity_dir_x": None, "vehicle_velocity_dir_y": None,
      "lookahead_vector_world_x": None, "lookahead_vector_world_y": None,
      "lookahead_dot_velocity": None, "lookahead_cross_velocity": None,
      "reference_lane_index": reference_lane_index, "current_lane_index": lane_idx,
      "metadrive_on_lane": bool(on_lane), "reference_road_id": None,
      "route_progress_m": None, "lateral_error_m": None, "heading_error_rad": None,
      "lane_width_m": None, "vehicle_width_m": vehicle_width,
      "traffic_vehicle_count": traffic_vehicle_count,
      "traffic_active_vehicle_count": len(active_traffic),
      "traffic_nearest_distance_m": nearest_distance,
      "traffic_nearest_closing_speed_mps": traffic_closing_speed,
      "traffic_nearest_ttc_s": traffic_ttc,
      "lane_departure": False, "collision": bool(getattr(vehicle, "crash_vehicle", False) or getattr(vehicle, "crash_object", False)),
      "specialist_teacher_curvature_1pm": None, "specialist_teacher_normalized_steer": None,
    }
    if lane is None:
      return result
    longitudinal, lateral = lane.local_coordinates(vehicle.position)
    lane_width = float(getattr(lane, "width", None) or vehicle_width)
    tangent = np.asarray(lane.position(longitudinal + 1.0, 0.0)) - np.asarray(lane.position(longitudinal - 1.0, 0.0))
    tangent_norm = float(np.linalg.norm(tangent))
    lookahead_m = float(simulator_control.get("lookahead_m", 0.0)) if simulator_control is not None else 0.0
    lookahead_vector = np.asarray(lane.position(longitudinal + lookahead_m, 0.0)) - np.asarray(vehicle.position)
    velocity_xy = np.asarray(vehicle.velocity[:2])
    velocity_xy_norm = float(np.linalg.norm(velocity_xy))
    velocity_direction = velocity_xy / velocity_xy_norm if velocity_xy_norm > 0.1 else None
    heading_error = (float(vehicle.heading_theta) - float(lane.heading_theta_at(longitudinal)) + math.pi) % (2 * math.pi) - math.pi
    result.update({
      "reference_road_id": str(getattr(lane, "index", (None,))[0]), "route_progress_m": float(longitudinal),
      "lateral_error_m": float(lateral), "heading_error_rad": heading_error, "lane_width_m": lane_width,
      "reference_curvature_1pm": ((float(lane.heading_theta_at(longitudinal + 1.0)) - float(lane.heading_theta_at(longitudinal - 1.0)) + math.pi) % (2 * math.pi) - math.pi) / 2.0,
      "lane_departure": abs(float(lateral)) > max(0.0, (lane_width - result["vehicle_width_m"]) / 2),
    })
    if specialist_dataset is not None:
      teacher = specialist_dataset["teacher"]
      target = lane.position(longitudinal + float(teacher["lookahead_m"]), 0.0)
      delta = np.asarray(target) - np.asarray(vehicle.position)
      distance = max(float(np.linalg.norm(delta)), 0.1)
      if velocity_direction is not None:
        alpha = world_signed_angle(velocity_direction, delta)
        teacher_curvature = 2.0 * math.sin(alpha) / distance
        result["specialist_teacher_curvature_1pm"] = teacher_curvature
        result["specialist_teacher_normalized_steer"] = float(np.clip(
          teacher_curvature * float(teacher["curvature_to_steer_gain"]), -0.2, 0.2))
    if tangent_norm > 0.0:
      result["reference_tangent_world_x"] = float(tangent[0] / tangent_norm)
      result["reference_tangent_world_y"] = float(tangent[1] / tangent_norm)
    if velocity_direction is not None:
      result.update({
        "vehicle_velocity_dir_x": float(velocity_direction[0]), "vehicle_velocity_dir_y": float(velocity_direction[1]),
        "lookahead_vector_world_x": float(lookahead_vector[0]), "lookahead_vector_world_y": float(lookahead_vector[1]),
        "lookahead_dot_velocity": float(np.dot(velocity_direction, lookahead_vector)),
        "lookahead_cross_velocity": float(velocity_direction[0] * lookahead_vector[1] - velocity_direction[1] * lookahead_vector[0]),
      })
    return result

  def simulator_controller(vehicle):
    nonlocal controller_target_curvature, controller_lookahead_heading_error
    assert simulator_control is not None
    lanes = list(getattr(vehicle.navigation, "current_ref_lanes", []) or [])
    lane = lanes[reference_lane_index] if 0 <= reference_lane_index < len(lanes) else None
    if lane is None:
      return 0.0, 0.0
    longitudinal, lateral = lane.local_coordinates(vehicle.position)
    if simulator_control["mode"] == "pure_pursuit":
      target = lane.position(longitudinal + float(simulator_control["lookahead_m"]), 0.0)
      delta = np.asarray(target) - np.asarray(vehicle.position)
      distance = max(float(np.linalg.norm(delta)), 0.1)
      velocity_xy = np.asarray(vehicle.velocity[:2])
      velocity_norm = float(np.linalg.norm(velocity_xy))
      if velocity_norm <= 0.1:
        steer = 0.0
      else:
        forward = velocity_xy / velocity_norm
        alpha = world_signed_angle(forward, delta)
        controller_target_curvature = 2.0 * math.sin(alpha) / distance
        controller_lookahead_heading_error = alpha
        steer = controller_target_curvature * float(simulator_control["curvature_to_steer_gain"])
    elif simulator_control["mode"] == "reference_curvature_follow":
      reference_curvature = ((float(lane.heading_theta_at(longitudinal + 1.0)) - float(lane.heading_theta_at(longitudinal - 1.0)) + math.pi) % (2 * math.pi) - math.pi) / 2.0
      velocity_xy = np.asarray(vehicle.velocity[:2])
      velocity_norm = float(np.linalg.norm(velocity_xy))
      tangent = np.asarray(lane.position(longitudinal + 1.0, 0.0)) - np.asarray(lane.position(longitudinal - 1.0, 0.0))
      tangent_heading = math.atan2(float(tangent[1]), float(tangent[0]))
      heading_error = 0.0 if velocity_norm <= 0.1 else world_signed_angle(velocity_xy / velocity_norm, np.array([math.cos(tangent_heading), math.sin(tangent_heading)]))
      controller_target_curvature = reference_curvature - float(simulator_control["lateral_gain"]) * float(lateral) - float(simulator_control["heading_gain"]) * heading_error
      controller_lookahead_heading_error = heading_error
      steer = controller_target_curvature * float(simulator_control["curvature_to_steer_gain"])
    else:
      target_heading = float(lane.heading_theta_at(longitudinal + float(simulator_control["lookahead_m"])))
      heading_error = (float(vehicle.heading_theta) - target_heading + math.pi) % (2 * math.pi) - math.pi
      steer = -float(simulator_control["lateral_gain"]) * float(lateral) - float(simulator_control["heading_gain"]) * heading_error
    speed_error = float(simulator_control["target_speed_mps"]) - float(np.linalg.norm(vehicle.velocity))
    return float(np.clip(steer, -0.2, 0.2)), float(np.clip(0.25 * speed_error, -1.0, 1.0))

  def specialist_controller(image):
    nonlocal specialist_prediction, specialist_last_image
    if image is None:
      specialist_prediction = 0.0
    elif image is not specialist_last_image:
      specialist_prediction = specialist_replay.predict(image)
      specialist_last_image = image
    speed_error = float(specialist_replay_config["target_speed_mps"]) - float(np.linalg.norm(env.vehicle.velocity))
    return specialist_prediction, float(np.clip(0.25 * speed_error, -1.0, 1.0))

  def reset():
    nonlocal previous_velocity_heading, previous_simulation_time_s, specialist_last_image, lead_vehicle, lead_visual_proxy
    # The fixed seed is configured as MetaDrive's start_seed. Passing it to
    # reset() would be interpreted as a bounded scenario index on 0.4.2.3.
    env.reset()
    lead_vehicle = lead_visual_proxy = None
    lead_config = environment_config.get("lead_vehicle")
    if lead_config is not None:
      from metadrive.component.vehicle.vehicle_type import StaticDefaultVehicle
      lanes = list(getattr(env.vehicle.navigation, "current_ref_lanes", []) or [])
      lane = lanes[reference_lane_index] if 0 <= reference_lane_index < len(lanes) else None
      if lane is None:
        raise RuntimeError("lead vehicle spawn requires the configured reference lane")
      longitudinal, _ = lane.local_coordinates(env.vehicle.position)
      lead_vehicle = env.engine.spawn_object(StaticDefaultVehicle, vehicle_config={
        "spawn_lane_index": lane.index, "spawn_longitude": longitudinal + float(lead_config["gap_m"]),
        "render_vehicle": bool(lead_config.get("render_vehicle", False)), "enable_reverse": False,
      })
      if lead_config.get("visual_proxy") == "box":
        from metadrive.engine.asset_loader import AssetLoader
        lead_visual_proxy = env.engine.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
        lead_visual_proxy.reparentTo(lead_vehicle.origin)
        lead_visual_proxy.setPos(0, 0, 1.0)
        lead_visual_proxy.setScale(1.0, 2.3, 0.8)
    previous_velocity_heading = None
    previous_simulation_time_s = None
    specialist_last_image = None
    if specialist_replay is not None:
      specialist_replay.reset()
    env.vehicle.config["max_speed_km_h"] = 1000
    lane_idx_prev, _ = get_current_lane_info(env.vehicle)

    simulation_state = metadrive_simulation_state(
      running=True,
      done=False,
      done_info=None,
    )
    simulation_state_send.send(simulation_state)

    return lane_idx_prev

  lane_idx_prev = reset()
  env.engine.sensors["rgb_road"].get_lens().setFov(camera_fov_deg)
  start_time = None
  previous_speed = 0.0

  def visual_proxy_bbox(camera):
    if lead_visual_proxy is None:
      return None
    bounds = lead_visual_proxy.getTightBounds()
    if bounds is None:
      return None
    lower, upper = bounds
    projected = []
    for x in (lower.x, upper.x):
      for y in (lower.y, upper.y):
        for z in (lower.z, upper.z):
          world = env.engine.render.getRelativePoint(lead_visual_proxy, Point3(x, y, z))
          camera_point = camera.getRelativePoint(env.engine.render, world)
          screen = Point2()
          if camera.node().getLens().project(camera_point, screen):
            projected.append(((screen.x + 1.0) * W / 2.0, (screen.y + 1.0) * H / 2.0))
    if not projected:
      return None
    xs, ys = zip(*projected)
    return [float(max(0.0, min(xs))), float(max(0.0, min(ys))),
            float(min(float(W), max(xs))), float(min(float(H), max(ys)))]

  def get_cam_as_rgb(cam):
    nonlocal debug_camera_captured, debug_capture_index
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(camera_position)
    cam.get_cam().setHpr(camera_hpr)
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get() # convert cupy array to numpy
    if camera_gamma != 1.0:
      img = np.rint(np.power(img.astype(np.float32) / 255.0, camera_gamma) * 255.0).astype(np.uint8)
    img = apply_camera_color_affine(img, camera_color_affine)
    if cam is env.engine.sensors["rgb_road"]:
      capture_path = metadata_path = None
      if debug_capture_index < len(debug_capture_frames) and rk.frame >= debug_capture_frames[debug_capture_index]:
        capture_path = os.path.join(debug_capture_dir, f"road-frame-{rk.frame:06d}.png")
        metadata_path = f"{capture_path}.json"
        debug_capture_index += 1
      elif debug_camera_path and not debug_camera_captured and rk.frame >= debug_camera_after_frame:
        capture_path, metadata_path = debug_camera_path, debug_camera_metadata_path
        debug_camera_captured = True
      if capture_path:
        os.makedirs(os.path.dirname(capture_path), exist_ok=True)
        Image.fromarray(img).save(capture_path)
        if metadata_path:
          os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
          with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump({"simulation_frame": rk.frame, "simulation_time_s": rk.frame / 100,
                       "camera_fov_deg": camera_fov_deg, "camera_focal_length_px": W / (2.0 * math.tan(math.radians(camera_fov_deg) / 2.0)),
                       "camera_position_m": list(map(float, camera_position)),
                       "camera_hpr_deg": list(map(float, camera_hpr)), "camera_gamma": camera_gamma,
                       "camera_color_affine": camera_color_affine,
                       "static_obstacle_bbox_xyxy_px": visual_proxy_bbox(cam.get_cam())}, handle, sort_keys=True)
    return img

  rk = Ratekeeper(100, None)

  vc = [0,0]
  steer_angle = gas = 0.0
  latest_road_image = None
  specialist_last_image = None

  while not exit_event.is_set():
    speed_mps = float(np.linalg.norm(env.vehicle.velocity))
    acceleration_mps2 = (speed_mps - previous_speed) * 100
    previous_speed = speed_mps
    ground_truth = reference_lane_telemetry(env.vehicle, rk.frame, rk.frame / 100, acceleration_mps2)
    vehicle_state = metadrive_vehicle_state(
      velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
      position=env.vehicle.position,
      bearing=float(math.degrees(env.vehicle.heading_theta)),
      steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING,
      ground_truth=ground_truth,
    )
    vehicle_state_send.send(vehicle_state)

    should_reset = False
    if controls_recv.poll(0):
      while controls_recv.poll(0):
        steer_angle, gas, should_reset = controls_recv.recv()

      if should_reset:
        lane_idx_prev = reset()
        start_time = None

    if simulator_control is not None:
      steer_metadrive, gas = simulator_controller(env.vehicle)
    elif specialist_replay is not None:
      steer_metadrive, gas = specialist_controller(latest_road_image)
    else:
      steer_metadrive = np.clip(steer_angle / (env.vehicle.MAX_STEERING * configured_steer_ratio), -1, 1)
    normalized_steer = float(steer_metadrive)
    vc = [steer_metadrive, gas]

    is_engaged = op_engaged.is_set()
    if is_engaged and start_time is None:
      start_time = time.monotonic()

    if rk.frame % 5 == 0:
      _, _, terminated, _, _ = env.step(vc)
      timeout = True if start_time is not None and time.monotonic() - start_time >= test_duration else False
      lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
      out_of_lane = lane_idx_curr != lane_idx_prev or not on_lane
      lane_idx_prev = lane_idx_curr

      if terminated or ((out_of_lane or timeout) and test_run):
        if terminated:
          done_result = env.done_function("default_agent")
        elif out_of_lane:
          done_result = (True, {"out_of_lane" : True})
        elif timeout:
          done_result = (True, {"timeout" : True})

        simulation_state = metadrive_simulation_state(
          running=False,
          done=done_result[0],
          done_info=done_result[1],
        )
        simulation_state_send.send(simulation_state)

      if dual_camera:
        wide_road_image[...] = get_cam_as_rgb("rgb_wide")
      latest_road_image = get_cam_as_rgb("rgb_road")
      road_image[...] = latest_road_image
      image_lock.release()

    rk.keep_time()
