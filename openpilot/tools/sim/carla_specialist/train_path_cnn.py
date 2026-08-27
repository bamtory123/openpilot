#!/usr/bin/env python3
"""Train a CARLA-only RGB-to-future-path CNN.

The validation partition holds out the ClearSunset illumination condition.  It
is intentionally a simulator evaluation artefact, not an openpilot driving
model and it does not replace the production model.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PATH_DISTANCES_M = np.asarray((5.0, 10.0, 20.0, 30.0, 40.0), dtype=np.float32)


@dataclass(frozen=True)
class Sample:
  image: Path
  path_y_m: np.ndarray
  curvature_1pm: float
  weather: str


def load_samples(dataset: Path) -> list[Sample]:
  rows = [json.loads(line) for line in (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
  if len(rows) < 40:
    raise ValueError("at least 40 labelled frames are required")
  return [Sample(dataset / row["image"], np.asarray(row["future_path_m"], dtype=np.float32)[:, 1],
                 float(row["curvature_1pm"]), row.get("weather", "unknown")) for row in rows]


class PathDataset(Dataset):
  def __init__(self, samples: list[Sample]):
    self.samples = samples

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    sample = self.samples[index]
    rgb = Image.open(sample.image).convert("RGB")
    # Preserve the horizon and the road.  This is deliberately independent of
    # openpilot's model input transform so the experiment remains contained.
    image = rgb.resize((160, 96), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(pixels), torch.from_numpy(sample.path_y_m), torch.tensor(sample.curvature_1pm, dtype=torch.float32)


class PathCNN(nn.Module):
  """A compact visual backbone with a five-point lateral-path head."""

  def __init__(self):
    super().__init__()
    self.backbone = nn.Sequential(
      nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
      nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
      nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
      nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
    )
    self.path_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(PATH_DISTANCES_M)))

  def forward(self, image: torch.Tensor) -> torch.Tensor:
    return self.path_head(self.backbone(image))


def curvature_from_path(path_y_m: torch.Tensor) -> torch.Tensor:
  # The CARLA route-curvature evaluator integrates a forward road segment.
  # The 30 m point matches that label contract much better than the nearly
  # straight 10 m point at curve entry.  The full five-point path remains the
  # primary target.
  return 2.0 * path_y_m[:, 3] / (PATH_DISTANCES_M[3] ** 2)


def metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
  model.eval()
  path_errors, curvature_errors, direction = [], [], []
  with torch.no_grad():
    for image, path_y, curvature in loader:
      predicted = model(image.to(device)).cpu()
      predicted_curvature = curvature_from_path(predicted)
      path_errors.append(torch.abs(predicted - path_y).numpy())
      curvature_errors.append(torch.abs(predicted_curvature - curvature).numpy())
      direction.append((predicted_curvature * curvature > 0.0).numpy())
  return {
    "path_lateral_mae_m": float(np.concatenate(path_errors).mean()),
    "curvature_mae_1pm": float(np.concatenate(curvature_errors).mean()),
    "curvature_direction_match": float(np.concatenate(direction).mean()),
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", type=Path)
  parser.add_argument("--output", type=Path, required=True, help="TorchScript .pt output")
  parser.add_argument("--epochs", type=int, default=35)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=20260823)
  parser.add_argument("--validation-weather", default="ClearSunset")
  args = parser.parse_args()
  if args.epochs < 1 or args.batch_size < 1:
    parser.error("epochs and batch-size must be positive")
  random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
  samples = load_samples(args.dataset)
  validation = [sample for sample in samples if sample.weather == args.validation_weather]
  training = [sample for sample in samples if sample.weather != args.validation_weather]
  if len(training) < 20 or len(validation) < 10:
    raise ValueError("need at least 20 training and 10 held-out-weather validation samples")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_loader = DataLoader(PathDataset(training), batch_size=args.batch_size, shuffle=True)
  validation_loader = DataLoader(PathDataset(validation), batch_size=args.batch_size)
  model = PathCNN().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
  path_loss = nn.SmoothL1Loss()
  for epoch in range(args.epochs):
    model.train()
    for image, target_path, target_curvature in train_loader:
      predicted_path = model(image.to(device))
      loss = path_loss(predicted_path / 10.0, target_path.to(device) / 10.0)
      loss = loss + 2.0 * path_loss(curvature_from_path(predicted_path), target_curvature.to(device))
      optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
      print(f"epoch={epoch + 1} train_loss={loss.item():.6f}")
  model.cpu().eval()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.jit.script(model).save(str(args.output))
  report = {
    "model": "carla_path_cnn_v1",
    "dataset": str(args.dataset),
    "train_samples": len(training),
    "validation_samples": len(validation),
    "validation_partition": f"weather != {args.validation_weather} / weather == {args.validation_weather}",
    "epochs": args.epochs,
    "input_rgb": [160, 96],
    "path_distances_m": PATH_DISTANCES_M.tolist(),
    "device": str(device),
    "train": metrics(model, DataLoader(PathDataset(training), batch_size=args.batch_size), torch.device("cpu")),
    "validation": metrics(model, validation_loader, torch.device("cpu")),
  }
  args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
