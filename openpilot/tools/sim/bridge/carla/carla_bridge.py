from __future__ import annotations

import math
from multiprocessing import Queue

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.carla.carla_world import CarlaWorld


class CarlaBridge(SimulatorBridge):
  """Bridge adapter; CARLA never modifies openpilot control authority."""

  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, *, host: str, port: int, town: str,
               route_asset: str, test_duration=math.inf, test_run=False, simlab_config=None):
    super().__init__(dual_camera, high_quality, simlab_config)
    self.host, self.port, self.town = host, port, town
    self.route_asset = route_asset
    self.test_duration = test_duration if test_run else math.inf
    self.test_run = test_run

  def spawn_world(self, queue: Queue):
    return CarlaWorld(queue, dual_camera=self.dual_camera, host=self.host, port=self.port,
                      town=self.town, route_asset=self.route_asset,
                      test_duration=self.test_duration, test_run=self.test_run)
