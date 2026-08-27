"""Small dependency-free image-to-curvature model used only in CARLA experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


FEATURE_WIDTH = 48
FEATURE_HEIGHT = 24


def image_features(rgb: np.ndarray) -> np.ndarray:
  """Return normalized road-view luminance and horizontal-gradient features."""
  if rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError(f"expected HxWx3 RGB image, got {rgb.shape}")
  # The upper half is primarily sky in the CARLA windshield camera. Keep the
  # road, lane markings, and near-horizon geometry while remaining compact.
  cropped = rgb[rgb.shape[0] // 2:, :, :]
  gray = Image.fromarray(cropped).convert("L").resize((FEATURE_WIDTH, FEATURE_HEIGHT), Image.Resampling.BILINEAR)
  values = np.asarray(gray, dtype=np.float32) / 255.0
  gradient = np.diff(values, axis=1, prepend=values[:, :1])
  return np.concatenate((values.ravel(), gradient.ravel())).astype(np.float32)


class SpecialistModel:
  """Ridge regressor serialized as a portable NumPy .npz artifact."""

  def __init__(self, feature_mean: np.ndarray, feature_scale: np.ndarray, weights: np.ndarray, bias: float):
    self.feature_mean = feature_mean.astype(np.float32)
    self.feature_scale = np.maximum(feature_scale.astype(np.float32), 1e-6)
    self.weights = weights.astype(np.float32)
    self.bias = float(bias)

  def predict(self, rgb: np.ndarray) -> float:
    features = image_features(rgb)
    return float(np.dot((features - self.feature_mean) / self.feature_scale, self.weights) + self.bias)

  @classmethod
  def load(cls, path: str | Path) -> "SpecialistModel":
    with np.load(path) as artifact:
      return cls(artifact["feature_mean"], artifact["feature_scale"], artifact["weights"], float(artifact["bias"]))

  def save(self, path: str | Path) -> None:
    np.savez_compressed(path, feature_mean=self.feature_mean, feature_scale=self.feature_scale, weights=self.weights, bias=self.bias)
