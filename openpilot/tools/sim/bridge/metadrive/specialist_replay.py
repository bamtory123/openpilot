from collections import deque

import numpy as np


FEATURE_HEIGHT = 24
FEATURE_WIDTH = 32
ARTIFACT_VERSION = 1
TEMPORAL_ARTIFACT_VERSION = 2


def image_features(image):
  if image.ndim != 3 or image.shape[2] != 3:
    raise ValueError("expected an RGB image")
  y = np.linspace(0, image.shape[0] - 1, FEATURE_HEIGHT, dtype=np.intp)
  x = np.linspace(0, image.shape[1] - 1, FEATURE_WIDTH, dtype=np.intp)
  rgb = image[np.ix_(y, x)].astype(np.float64)
  return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).reshape(-1) / 255.0


class SpecialistReplay:
  def __init__(self, artifact_path):
    with np.load(artifact_path, allow_pickle=False) as artifact:
      self.version = int(artifact["version"])
      if self.version not in (ARTIFACT_VERSION, TEMPORAL_ARTIFACT_VERSION):
        raise ValueError("unsupported specialist replay artifact")
      self.mean = artifact["mean"]
      self.scale = artifact["scale"]
      self.weights = artifact["weights"]
      self.frame_gap = int(artifact["frame_gap"]) if self.version == TEMPORAL_ARTIFACT_VERSION else 0
    expected_feature_count = FEATURE_HEIGHT * FEATURE_WIDTH * (2 if self.version == TEMPORAL_ARTIFACT_VERSION else 1)
    if self.mean.shape != (expected_feature_count,) or self.scale.shape != self.mean.shape or self.weights.shape != (len(self.mean) + 1,):
      raise ValueError("invalid specialist replay artifact shape")
    self.history = deque(maxlen=(self.frame_gap // 5) + 1) if self.version == TEMPORAL_ARTIFACT_VERSION else None

  def predict(self, image):
    current_features = image_features(image)
    if self.history is not None:
      self.history.append(current_features)
      previous_features = self.history[0]
      current_features = np.concatenate((current_features, current_features - previous_features))
    features = (current_features - self.mean) / self.scale
    return float(np.clip(np.append(features, 1.0) @ self.weights, -0.2, 0.2))
