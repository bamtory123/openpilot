#!/usr/bin/env python3
"""Train a simulator-only RGB-to-curvature ridge model from CARLA labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from openpilot.tools.sim.carla_specialist.model import SpecialistModel, image_features


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
  labels_path = dataset_dir / "labels.jsonl"
  rows = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line]
  if len(rows) < 20:
    raise ValueError("at least 20 labelled frames are required")
  features = [image_features(np.asarray(Image.open(dataset_dir / row["image"]).convert("RGB"))) for row in rows]
  labels = [float(row["curvature_1pm"]) for row in rows]
  return np.stack(features), np.asarray(labels, dtype=np.float32)


def fit_ridge(features: np.ndarray, labels: np.ndarray, alpha: float) -> SpecialistModel:
  mean = features.mean(axis=0)
  scale = features.std(axis=0)
  normalized = (features - mean) / np.maximum(scale, 1e-6)
  centered_labels = labels - labels.mean()
  # Dual form is stable for the small CARLA dataset and avoids a large
  # feature-by-feature matrix inversion.
  kernel = normalized @ normalized.T
  dual = np.linalg.solve(kernel + alpha * np.eye(len(features), dtype=np.float32), centered_labels)
  weights = normalized.T @ dual
  return SpecialistModel(mean, scale, weights, float(labels.mean()))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--alpha", type=float, default=10.0)
  args = parser.parse_args()
  if args.alpha <= 0:
    parser.error("alpha must be positive")
  features, labels = load_dataset(args.dataset)
  split = max(1, int(len(labels) * 0.8))
  model = fit_ridge(features[:split], labels[:split], args.alpha)
  predictions = np.asarray([model.predict(np.asarray(Image.open(args.dataset / json.loads(line)["image"]).convert("RGB")))
                            for line in (args.dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines()[split:] if line])
  validation = labels[split:]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  model.save(args.output)
  report = {
    "train_samples": split,
    "validation_samples": len(validation),
    "validation_mae_1pm": float(np.mean(np.abs(predictions - validation))) if len(validation) else None,
    "validation_direction_match": float(np.mean(predictions * validation > 0.0)) if len(validation) else None,
    "alpha": args.alpha,
  }
  args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(report, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
