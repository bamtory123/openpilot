import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from openpilot.tools.sim.bridge.carla.control import normalized_steer
from openpilot.tools.sim.bridge.carla.optics import narrow_road_fov_deg
from openpilot.tools.sim.bridge.carla.route_asset import load_route_asset


def test_normalized_steer_clamps_at_wheel_limit():
  assert normalized_steer(0.0, 100.0) == 0.0
  assert normalized_steer(50.0, 100.0) == 0.5
  assert normalized_steer(-200.0, 100.0) == -1.0


def test_carla_interface_uses_opposite_steering_sign():
  assert normalized_steer(-70.0, 700.0) == -0.1


def test_normalized_steer_rejects_invalid_limit():
  with pytest.raises(ValueError):
    normalized_steer(1.0, 0.0)


def test_narrow_road_fov_matches_c3_optics_contract():
  assert 39.9 < narrow_road_fov_deg(1928) < 40.1


def test_route_asset_is_immutable_and_hashed(tmp_path: Path):
  path = tmp_path / "town04.json"
  path.write_text(json.dumps({"town": "Town04", "turn_plan": ["straight", "left"],
                               "points": [[float(i), 0, 0, 0, 0, 0] for i in range(80)]}), encoding="utf-8")
  route = load_route_asset(path, expected_town="Town04")
  assert len(route.points) == 80
  assert len(route.sha256) == 64
  with pytest.raises(FrozenInstanceError):
    route.town = "Town03"  # type: ignore[misc]


def test_route_asset_rejects_wrong_town(tmp_path: Path):
  path = tmp_path / "town03.json"
  path.write_text(json.dumps({"town": "Town03", "points": [[0, 0, 0, 0, 0, 0]] * 80}), encoding="utf-8")
  with pytest.raises(RuntimeError, match="town mismatch"):
    load_route_asset(path, expected_town="Town04")
