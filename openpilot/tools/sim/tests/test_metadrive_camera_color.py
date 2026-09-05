import numpy as np
from panda3d.core import Texture

from openpilot.tools.sim.bridge.metadrive.metadrive_common import apply_camera_color_affine, apply_texture_luma_gain


def test_camera_color_affine_preserves_identity_and_clips_output():
  image = np.asarray([[[10, 20, 250]]], dtype=np.uint8)

  assert apply_camera_color_affine(image, None) is image
  transformed = apply_camera_color_affine(image, {"gain_rgb": [2.0, 1.0, 1.0], "bias_rgb": [0.0, -30.0, 20.0]})
  np.testing.assert_array_equal(transformed, np.asarray([[[20, 0, 255]]], dtype=np.uint8))


def test_texture_luma_gain_preserves_identity_and_scales_native_rgb_bytes():
  texture = Texture("fixture")
  texture.setup2dTexture(2, 1, Texture.TUnsignedByte, Texture.F_rgb)
  texture.setRamImage(np.asarray([10, 20, 30, 100, 150, 200], dtype=np.uint8))

  assert apply_texture_luma_gain(texture, 1.0) is texture
  before = np.frombuffer(texture.getRamImage().getData(), dtype=np.uint8).copy()
  assert apply_texture_luma_gain(texture, 0.75) is texture
  np.testing.assert_array_equal(np.frombuffer(texture.getRamImage().getData(), dtype=np.uint8), np.rint(before * 0.75).astype(np.uint8))
