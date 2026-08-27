import math
from multiprocessing import Queue

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.carla.carla_world import CarlaWorld


class CarlaBridge(SimulatorBridge):
  """CARLA implementation of openpilot's existing simulator bridge contract."""

  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera: bool, high_quality: bool, *, host: str = "127.0.0.1", port: int = 2000,
               town: str = "Town04", scenario: str = "curve_60s", traffic_count: int = 0,
               test_duration: float = math.inf, test_run: bool = False):
    super().__init__(dual_camera, high_quality)
    self.host = host
    self.port = port
    self.town = town
    self.scenario = scenario
    self.traffic_count = traffic_count
    self.test_run = test_run
    self.test_duration = test_duration if test_run else math.inf

  def spawn_world(self, queue: Queue):
    return CarlaWorld(
      queue, host=self.host, port=self.port, town=self.town, scenario=self.scenario, dual_camera=self.dual_camera,
      traffic_count=self.traffic_count, test_duration=self.test_duration, test_run=self.test_run,
    )
