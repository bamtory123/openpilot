import unittest

import numpy as np

from openpilot.tools.sim.lib.camerad import rgb_to_nv12


class TestCamerad(unittest.TestCase):
  def test_rgb_to_nv12_black_frame_has_limited_range_black_and_neutral_chroma(self):
    nv12 = np.frombuffer(rgb_to_nv12(np.zeros((4, 4, 3), dtype=np.uint8)), dtype=np.uint8)

    self.assertEqual(nv12.size, 4 * 4 * 3 // 2)
    self.assertTrue(np.all(nv12[:16] == 16))
    self.assertTrue(np.all(nv12[16:] == 128))

  def test_rgb_to_nv12_preserves_bt601_rgb_channel_order(self):
    expected = {
      "white": ((255, 255, 255), 237, (128, 128)),
      "red": ((255, 0, 0), 82, (109, 184)),
      "green": ((0, 255, 0), 145, (91, 81)),
      "blue": ((0, 0, 255), 42, (184, 119)),
    }
    for color, (rgb, y, uv) in expected.items():
      with self.subTest(color=color):
        frame = np.full((4, 4, 3), rgb, dtype=np.uint8)
        nv12 = np.frombuffer(rgb_to_nv12(frame), dtype=np.uint8)
        self.assertTrue(np.all(nv12[:16] == y))
        self.assertTrue(np.all(nv12[16::2] == uv[0]))
        self.assertTrue(np.all(nv12[17::2] == uv[1]))
