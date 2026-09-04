import numpy as np

from openpilot.tools.sim.bridge.metadrive.metadrive_common import apply_camera_color_affine


def test_camera_color_affine_preserves_identity_and_clips_output():
  image = np.asarray([[[10, 20, 250]]], dtype=np.uint8)

  assert apply_camera_color_affine(image, None) is image
  transformed = apply_camera_color_affine(image, {"gain_rgb": [2.0, 1.0, 1.0], "bias_rgb": [0.0, -30.0, 20.0]})
  np.testing.assert_array_equal(transformed, np.asarray([[[20, 0, 255]]], dtype=np.uint8))
