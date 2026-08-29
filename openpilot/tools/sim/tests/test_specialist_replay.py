import tempfile
import unittest

import numpy as np

from openpilot.tools.sim.bridge.metadrive.specialist_replay import FEATURE_HEIGHT, FEATURE_WIDTH, SpecialistReplay


class TestSpecialistReplay(unittest.TestCase):
  def test_replay_uses_rgb_image_and_clips_output(self):
    with tempfile.NamedTemporaryFile(suffix=".npz") as artifact:
      np.savez(artifact.name, version=np.array(1), mean=np.zeros(FEATURE_HEIGHT * FEATURE_WIDTH),
               scale=np.ones(FEATURE_HEIGHT * FEATURE_WIDTH), weights=np.r_[np.zeros(FEATURE_HEIGHT * FEATURE_WIDTH), 1.0])
      replay = SpecialistReplay(artifact.name)
      self.assertEqual(replay.predict(np.zeros((4, 4, 3), dtype=np.uint8)), 0.2)
