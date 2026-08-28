import math
import numpy as np
import unittest

from openpilot.tools.sim.bridge.metadrive.metadrive_process import world_signed_angle


class TestMetaDriveGeometry(unittest.TestCase):
  def test_world_signed_angle_uses_cross_product_sign(self):
    forward = np.array([1.0, 0.0])
    self.assertTrue(math.isclose(world_signed_angle(forward, np.array([1.0, 1.0])), math.pi / 4))
    self.assertTrue(math.isclose(world_signed_angle(forward, np.array([1.0, -1.0])), -math.pi / 4))

  def test_world_signed_angle_is_zero_for_aligned_vectors(self):
    self.assertTrue(math.isclose(world_signed_angle(np.array([0.0, 1.0]), np.array([0.0, 12.0])), 0.0))
