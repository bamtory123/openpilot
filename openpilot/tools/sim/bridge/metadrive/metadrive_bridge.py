import math
from multiprocessing import Queue

from metadrive.component.map.pg_map import MapGenerateMethod

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }

def create_map(track_size=60):
  curve_len = track_size * 2
  return {
    "type": MapGenerateMethod.PG_MAP_FILE,
    "lane_num": 2,
    "lane_width": 4.5,
    "config": [
      None,
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
      straight_block(track_size),
      curve_block(curve_len, 90),
    ]
  }


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False, simlab_config=None):
    super().__init__(dual_camera, high_quality, simlab_config)

    self.should_render = False
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = {
      "use_render": self.should_render,
      "vehicle_config": {
        "enable_reverse": False,
        "render_vehicle": False,
        "image_source": "rgb_road",
      },
      "sensors": sensors,
      # Keep simulator images in host memory. Under WSL, MetaDrive's CUDA image
      # path and tinygrad modeld sharing the RTX can poison the CUDA context.
      "image_on_cuda": False,
      "image_observation": True,
      "interface_panel": [],
      "out_of_route_done": False,
      "on_continuous_line_done": False,
      "crash_vehicle_done": False,
      "crash_object_done": False,
      "arrive_dest_done": False,
      "traffic_density": 0.0, # traffic is incredibly expensive
      "random_spawn_lane_index": False,
      "map_config": create_map(),
      "decision_repeat": 1,
      "physics_world_step_size": self.TICKS_PER_FRAME/100,
      "preload_models": False,
      "show_logo": False,
      "anisotropic_filtering": False
    }

    # Only fields implemented by the bridge are exposed to the simulator.
    config["simlab"] = self.simlab_config
    seed = self.simlab_config.get("environment", {}).get("seed")
    if seed is not None:
      # MetaDrive 0.4.2.3 treats reset(seed=...) as a scenario index and
      # rejects arbitrary integers. start_seed fixes the generated scenario
      # while preserving the normal reset lifecycle.
      config["start_seed"] = int(seed)
      config["num_scenarios"] = 1

    return MetaDriveWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)
