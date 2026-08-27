#!/usr/bin/env python3
"""Collect stable CARLA RGB, future-path, and lane-centre labels for specialist training."""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
from pathlib import Path

import numpy as np
from PIL import Image

from openpilot.tools.sim.bridge.carla.carla_world import CAMERA_HORIZONTAL_FOV_DEG, CAMERA_MOUNT
from openpilot.tools.sim.lib.common import H, W

PATH_DISTANCES_M = (5.0, 10.0, 20.0, 30.0, 40.0)
WEATHER_PRESETS = ("ClearNoon", "CloudyNoon", "WetCloudyNoon", "ClearSunset")


def yaw_delta_deg(before: float, after: float) -> float:
  return (after - before + 180.0) % 360.0 - 180.0


def trace_route(waypoint, step_m: float = 2.0, distance_m: float = 46.0):
  route = [waypoint]
  for _ in range(int(distance_m / step_m)):
    choices = route[-1].next(step_m)
    if not choices:
      break
    yaw = route[-1].transform.rotation.yaw
    route.append(min(choices, key=lambda item: abs(yaw_delta_deg(yaw, item.transform.rotation.yaw))))
  return route


def route_labels(route, step_m: float = 2.0) -> tuple[float, list[list[float]]] | None:
  if len(route) < 22:
    return None
  ego = route[0].transform
  forward, right = ego.get_forward_vector(), ego.get_right_vector()
  path = []
  for distance_m in PATH_DISTANCES_M:
    point = route[round(distance_m / step_m)].transform.location
    dx, dy = point.x - ego.location.x, point.y - ego.location.y
    path.append([round(dx * forward.x + dy * forward.y, 4), round(-(dx * right.x + dy * right.y), 4)])
  curvature = -math.radians(yaw_delta_deg(route[2].transform.rotation.yaw, route[12].transform.rotation.yaw)) / (10.0 * step_m)
  return curvature, path


def newest_frame(frame_queue: queue.Queue):
  image = None
  while True:
    try:
      image = frame_queue.get_nowait()
    except queue.Empty:
      return image


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--town", default="Town04")
  parser.add_argument("--samples", type=int, default=200)
  parser.add_argument("--seed", type=int, default=20260822)
  parser.add_argument("--weather", choices=["clear", "mixed"], default="mixed")
  parser.add_argument("--min-abs-curvature", type=float, default=0.0,
                      help="reject nearly straight route segments; use 0.002 for curve-focused data")
  args = parser.parse_args()
  if args.samples < 20:
    parser.error("samples must be at least 20")
  if args.min_abs_curvature < 0.0:
    parser.error("min-abs-curvature must be non-negative")
  try:
    import carla
  except ImportError as exc:
    raise SystemExit("carla==0.9.16 is required in the openpilot venv") from exc
  if args.output.exists() and any(args.output.iterdir()):
    raise SystemExit(f"output directory must be empty: {args.output}")
  args.output.mkdir(parents=True, exist_ok=True)
  image_dir = args.output / "images"
  image_dir.mkdir()
  client = carla.Client(args.host, args.port)
  client.set_timeout(20.0)
  world = client.get_world()
  if world.get_map().name.rsplit("/", 1)[-1] != args.town:
    world = client.load_world(args.town)
  original = world.get_settings()
  settings = world.get_settings()
  settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
  world.apply_settings(settings)
  randomizer, spawn_points = random.Random(args.seed), world.get_map().get_spawn_points()
  bp_lib = world.get_blueprint_library()
  vehicle = world.try_spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0], spawn_points[0])
  if vehicle is None:
    raise RuntimeError("unable to spawn collector vehicle")
  vehicle.set_simulate_physics(False)
  camera_bp = bp_lib.find("sensor.camera.rgb")
  camera_bp.set_attribute("image_size_x", str(W)); camera_bp.set_attribute("image_size_y", str(H))
  camera_bp.set_attribute("fov", f"{CAMERA_HORIZONTAL_FOV_DEG:.6f}"); camera_bp.set_attribute("sensor_tick", "0.05")
  frame_queue: queue.Queue = queue.Queue(maxsize=3)
  camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=CAMERA_MOUNT[0], y=CAMERA_MOUNT[1], z=CAMERA_MOUNT[2]),
                                                          carla.Rotation(pitch=CAMERA_MOUNT[3])), attach_to=vehicle)
  camera.listen(lambda image: frame_queue.put_nowait(image) if not frame_queue.full() else None)
  written = 0
  try:
    with (args.output / "labels.jsonl").open("w", encoding="utf-8") as labels:
      attempts = 0
      while written < args.samples and attempts < args.samples * 12:
        attempts += 1
        waypoint = world.get_map().get_waypoint(randomizer.choice(spawn_points).location, project_to_road=True, lane_type=carla.LaneType.Driving)
        route = trace_route(waypoint) if waypoint is not None else []
        target = route_labels(route)
        if target is None:
          continue
        curvature, path = target
        if abs(curvature) < args.min_abs_curvature:
          continue
        weather = "ClearNoon" if args.weather == "clear" else WEATHER_PRESETS[(written * len(WEATHER_PRESETS)) // args.samples]
        world.set_weather(getattr(carla.WeatherParameters, weather))
        vehicle.set_transform(route[0].transform)
        image = None
        for _ in range(3):
          world.tick()
          image = newest_frame(frame_queue) or image
        if image is None:
          continue
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((H, W, 4))
        filename = f"images/{written:06d}.png"
        Image.fromarray(raw[:, :, :3][:, :, ::-1]).save(args.output / filename)
        start = route[0].transform.location
        labels.write(json.dumps({"image": filename, "curvature_1pm": curvature, "future_path_m": path,
                                 "lane_center_path_m": path, "weather": weather, "town": args.town, "seed": args.seed,
                                 "route_start_xy": [round(start.x, 2), round(start.y, 2)]}) + "\n")
        written += 1
  finally:
    camera.stop(); camera.destroy(); vehicle.destroy()
    try:
      world.tick(); world.apply_settings(original)
    except RuntimeError:
      pass
  manifest = {"samples": written, "requested_samples": args.samples, "town": args.town, "seed": args.seed,
              "weather": args.weather, "camera_mount": CAMERA_MOUNT, "camera_hfov_deg": CAMERA_HORIZONTAL_FOV_DEG,
              "future_path_distances_m": PATH_DISTANCES_M, "min_abs_curvature_1pm": args.min_abs_curvature}
  (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(manifest, indent=2))
  return 0 if written >= args.samples else 2


if __name__ == "__main__":
  raise SystemExit(main())
