#!/usr/bin/env python3
"""Render an inspectable camera/perception overlay from one CARLA harness run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


CURVES = (
  ("CARLA road reference", "carla_route_reference_curvature_1pm", (62, 220, 120)),
  ("stock model", "model_desired_curvature_1pm", (80, 160, 255)),
  ("specialist", "carla_specialist_curvature_1pm", (255, 205, 60)),
  ("actual vehicle", "carla_actual_curvature_1pm", (255, 95, 95)),
)


def latest_run(root: Path) -> Path:
  candidates = [path for path in root.iterdir() if path.is_dir() and (path / "run.json").is_file()]
  if not candidates:
    raise FileNotFoundError(f"no harness runs found in {root}")
  return max(candidates, key=lambda path: path.stat().st_mtime)


def finite(value) -> float | None:
  try:
    number = float(value)
  except (TypeError, ValueError):
    return None
  return number if abs(number) < 1e6 else None


def sample_for_frame(samples: list[dict], active_start: float, active_elapsed: float) -> dict:
  target = active_start + active_elapsed
  return min(samples, key=lambda sample: abs(float(sample["t_s"]) - target))


def draw_curve(draw: ImageDraw.ImageDraw, width: int, height: int, curvature: float, color: tuple[int, int, int]):
  # Camera-space qualitative path overlay. CARLA reference already uses the
  # openpilot lateral sign convention; negative curvature projects to image
  # right. It is intentionally labelled as a diagnostic projection, not a
  # calibrated lane segmentation mask.
  points = []
  for distance_m in range(0, 46, 2):
    y = height - 65 - distance_m * 12
    x = width / 2 - curvature * distance_m * distance_m * 420
    points.append((x, y))
  draw.line(points, fill=color, width=5)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run", type=Path, nargs="?")
  parser.add_argument("--latest-root", type=Path)
  args = parser.parse_args()
  if (args.run is None) == (args.latest_root is None):
    parser.error("provide exactly one of RUN or --latest-root")
  run_dir = args.run if args.run is not None else latest_run(args.latest_root)
  samples = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
  active = [sample for sample in samples if sample.get("active")]
  if not active:
    raise RuntimeError("run contains no active samples")
  active_start = float(active[0]["t_s"])
  frames = sorted((run_dir / "camera_frames").glob("active-*.png"))
  if not frames:
    raise RuntimeError("run contains no camera frames; rerun with --capture-camera-frames")
  annotated: list[Image.Image] = []
  for frame_path in frames:
    match = re.search(r"-(\d+(?:\.\d+)?)s\.png$", frame_path.name)
    if match is None:
      continue
    sample = sample_for_frame(samples, active_start, float(match.group(1)))
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, field, color in CURVES:
      value = finite(sample.get(field))
      if value is not None:
        draw_curve(draw, image.width, image.height, value, color)
    draw.rectangle((16, 16, 550, 170), fill=(0, 0, 0))
    draw.text((28, 28), f"active {float(match.group(1)):.1f}s | source: {sample.get('carla_control_source', 'openpilot')}", fill=(255, 255, 255))
    y = 55
    for label, field, color in CURVES:
      value = finite(sample.get(field))
      draw.text((28, y), f"{label}: {value if value is not None else 0.0:+.4f} 1/m", fill=color)
      y += 25
    annotated.append(image.resize((640, 402)))
  if not annotated:
    raise RuntimeError("no camera frames could be annotated")
  sheet = Image.new("RGB", (640, 402 * len(annotated)), color=(20, 20, 20))
  for index, image in enumerate(annotated):
    sheet.paste(image, (0, index * 402))
  output = run_dir / "perception-report.png"
  sheet.save(output)
  print(output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
