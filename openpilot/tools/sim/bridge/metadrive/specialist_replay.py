import numpy as np


FEATURE_HEIGHT = 24
FEATURE_WIDTH = 32
ARTIFACT_VERSION = 1


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
      if int(artifact["version"]) != ARTIFACT_VERSION:
        raise ValueError("unsupported specialist replay artifact")
      self.mean = artifact["mean"]
      self.scale = artifact["scale"]
      self.weights = artifact["weights"]
    if self.mean.shape != (FEATURE_HEIGHT * FEATURE_WIDTH,) or self.scale.shape != self.mean.shape or self.weights.shape != (len(self.mean) + 1,):
      raise ValueError("invalid specialist replay artifact shape")

  def predict(self, image):
    features = (image_features(image) - self.mean) / self.scale
    return float(np.clip(np.append(features, 1.0) @ self.weights, -0.2, 0.2))
