#!/usr/bin/env python3
"""Measure CARLA Tesla steering response without running openpilot.

This is a simulator-physics calibration, not a real-vehicle calibration. It
maps CARLA's normalized VehicleControl.steer input to observed yaw rate so that
the bridge mapping can be reviewed independently of the openpilot controller.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def yaw_delta_deg(before: float, after: float) -> float:
  return (after - before + 180.0) % 360.0 - 180.0


def select_straight_spawn(carla_map, spawn_points, lane_type):
  """Choose a deterministic start with the least heading change over 40 m."""
  best = spawn_points[0]
  best_score = float("inf")
  for spawn in spawn_points[::max(1, len(spawn_points) // 30)]:
    waypoint = carla_map.get_waypoint(spawn.location, project_to_road=True, lane_type=lane_type)
    if waypoint is None:
      continue
    score = 0.0
    for _ in range(20):
      next_points = waypoint.next(2.0)
      if not next_points:
        score = float("inf")
        break
      following = min(next_points, key=lambda point: abs(yaw_delta_deg(waypoint.transform.rotation.yaw, point.transform.rotation.yaw)))
      score += abs(yaw_delta_deg(waypoint.transform.rotation.yaw, following.transform.rotation.yaw))
      waypoint = following
    if score < best_score:
      best, best_score = spawn, score
  return best, best_score


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--speed-mps", type=float, default=8.0)
  parser.add_argument("--seconds", type=float, default=2.0)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  import carla

  client = carla.Client(args.host, args.port)
  client.set_timeout(30.0)
  world = client.get_world()
  original_settings = world.get_settings()
  settings = world.get_settings()
  settings.synchronous_mode = True
  settings.fixed_delta_seconds = 0.05
  world.apply_settings(settings)

  vehicle = None
  try:
    spawn, straightness_score = select_straight_spawn(world.get_map(), world.get_map().get_spawn_points(), carla.LaneType.Driving)
    vehicle_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    wheel_limits = []
    samples = []
    steps = max(1, round(args.seconds / settings.fixed_delta_seconds))
    for steer in (-0.05, -0.02, 0.02, 0.05):
      if vehicle is not None:
        vehicle.destroy()
        world.tick()
      vehicle = world.try_spawn_actor(vehicle_bp, spawn)
      if vehicle is None:
        raise RuntimeError("unable to spawn calibration vehicle")
      if not wheel_limits:
        wheel_limits = [wheel.max_steer_angle for wheel in vehicle.get_physics_control().wheels if wheel.max_steer_angle > 0]
      forward = spawn.get_forward_vector()
      vehicle.set_target_velocity(carla.Vector3D(
        x=args.speed_mps * forward.x, y=args.speed_mps * forward.y, z=0.0,
      ))
      for _ in range(4):
        world.tick()
      start_yaw = vehicle.get_transform().rotation.yaw
      vehicle.apply_control(carla.VehicleControl(steer=steer))
      for _ in range(steps):
        world.tick()
      end_yaw = vehicle.get_transform().rotation.yaw
      velocity = vehicle.get_velocity()
      speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
      yaw_rate_degps = yaw_delta_deg(start_yaw, end_yaw) / (steps * settings.fixed_delta_seconds)
      samples.append({
        "normalized_steer": steer,
        "speed_mps": round(speed, 4),
        "yaw_rate_degps": round(yaw_rate_degps, 4),
      })

    result = {
      "server_version": client.get_server_version(),
      "map": world.get_map().name,
      "fixed_delta_seconds": settings.fixed_delta_seconds,
      "requested_speed_mps": args.speed_mps,
      "straightness_score_deg": round(straightness_score, 4),
      "max_front_wheel_deg": max(wheel_limits) if wheel_limits else None,
      "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0
  finally:
    if vehicle is not None:
      vehicle.destroy()
    world.apply_settings(original_settings)


if __name__ == "__main__":
  raise SystemExit(main())
