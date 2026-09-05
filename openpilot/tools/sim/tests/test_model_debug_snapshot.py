from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.common import build_model_debug_snapshot


def _points():
  return SimpleNamespace(x=[5.0, 20.0], y=[0.0, 0.1], z=[0.0, 0.0])


def test_model_debug_snapshot_preserves_geometry_and_projection_contract():
  model = SimpleNamespace(frameId=42, position=_points(), laneLines=[_points()] * 4,
                          laneLineProbs=[0.1, 0.8, 0.7, 0.1])
  calibration = SimpleNamespace(rpyCalib=[0.01, -0.02, 0.03], height=[1.22])
  camera = SimpleNamespace(narrow_road=SimpleNamespace(intrinsics=np.eye(3)))

  snapshot = build_model_debug_snapshot(model, calibration, camera, 100, 102)

  assert snapshot["scope"] == "analysis_only_model_projection_not_runtime_control_or_accuracy"
  assert snapshot["requested_camera_source_frame_id"] == 100
  assert snapshot["bridge_frame"] == 102
  assert snapshot["model_frame_id"] == 42
  assert snapshot["path"][1] == [20.0, 0.1, 0.0]
  assert snapshot["lane_line_probabilities"][1:3] == [0.8, 0.7]
  assert snapshot["projection"]["camera_height_m"] == 1.22
