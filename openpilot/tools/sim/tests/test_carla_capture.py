import json

import numpy as np

from openpilot.tools.sim.bridge.carla.capture import CaptureWriter


def test_capture_writer_saves_immutable_frame(tmp_path):
  events = []
  writer = CaptureWriter(str(tmp_path), 2, events.append)
  frame = np.zeros((2, 3, 3), dtype=np.uint8)
  frame[0, 0] = (10, 20, 30)
  writer.offer(4, 123, frame)
  frame[0, 0] = (0, 0, 0)

  assert writer.close() == 0
  assert (tmp_path / "road-frame-000004.png").is_file()
  assert json.loads((tmp_path / "road-frame-000004.json").read_text())["capture_mono_ns"] == 123
  assert events[0]["image"] == "captures/road-frame-000004.png"


def test_capture_writer_samples_configured_frames(tmp_path):
  writer = CaptureWriter(str(tmp_path), 3, lambda _: None)
  writer.offer(4, 1, np.zeros((1, 1, 3), dtype=np.uint8))
  assert writer.close() == 0
  assert not list(tmp_path.glob("*.png"))
