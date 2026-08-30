import warnings
import unittest
import importlib

# Since metadrive depends on pkg_resources, and pkg_resources is deprecated as an API
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
  bridge_module = importlib.import_module("openpilot.tools.sim.bridge.metadrive.metadrive_bridge")
  MetaDriveBridge = bridge_module.MetaDriveBridge
  create_map = bridge_module.create_map
except ModuleNotFoundError:
  MetaDriveBridge = None
  create_map = None
from openpilot.tools.sim.tests.test_sim_bridge import TestSimBridgeBase

@unittest.skipIf(MetaDriveBridge is None, "metadrive is not installed")
class TestMetaDriveBridge(TestSimBridgeBase):
  def setup_method(self):
    super().openpilot_setup_method()
    self.test_duration = 30

  def create_bridge(self):
    assert MetaDriveBridge is not None
    return MetaDriveBridge(False, False, self.test_duration, True)

  def test_serpentine_profile_alternates_curve_directions(self):
    assert create_map is not None
    curves = [block["dir"] for block in create_map(60, 1, "serpentine")["config"] if block and block["id"] == "C"]
    assert curves == [1, 0, 0, 1]

  def test_loop_profile_preserves_curve_direction(self):
    assert create_map is not None
    curves = [block["dir"] for block in create_map(60, 1)["config"] if block and block["id"] == "C"]
    assert curves == [1, 1, 1, 1]
