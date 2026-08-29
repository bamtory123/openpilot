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

  def test_temporal_replay_uses_delayed_rgb_difference(self):
    count = FEATURE_HEIGHT * FEATURE_WIDTH * 2
    with tempfile.NamedTemporaryFile(suffix=".npz") as artifact:
      weights = np.zeros(count + 1); weights[FEATURE_HEIGHT * FEATURE_WIDTH] = 1.0
      np.savez(artifact.name, version=np.array(2), frame_gap=np.array(20), mean=np.zeros(count), scale=np.ones(count), weights=weights)
      replay = SpecialistReplay(artifact.name)
      replay.predict(np.zeros((4, 4, 3), dtype=np.uint8))
      replay.predict(np.zeros((4, 4, 3), dtype=np.uint8))
      replay.predict(np.zeros((4, 4, 3), dtype=np.uint8))
      self.assertGreater(replay.predict(np.full((4, 4, 3), 255, dtype=np.uint8)), 0.2 - 1e-9)
