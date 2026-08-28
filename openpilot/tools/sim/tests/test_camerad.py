import unittest

import numpy as np

from openpilot.tools.sim.lib.camerad import rgb_to_nv12


class TestCamerad(unittest.TestCase):
  def test_rgb_to_nv12_black_frame_has_limited_range_black_and_neutral_chroma(self):
    nv12 = np.frombuffer(rgb_to_nv12(np.zeros((4, 4, 3), dtype=np.uint8)), dtype=np.uint8)

    self.assertEqual(nv12.size, 4 * 4 * 3 // 2)
    self.assertTrue(np.all(nv12[:16] == 16))
    self.assertTrue(np.all(nv12[16:] == 128))
