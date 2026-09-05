import json
import os
from pathlib import Path
import time

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from PIL import Image

from openpilot.common.realtime import DT_DMON
from openpilot.tools.sim.lib.camerad import Camerad
from openpilot.tools.sim.lib.camera_transport import CameraTransportDelay, QueuedCameraFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from openpilot.tools.sim.lib.common import World, SimulatorState


class SimulatedSensors:
  """Simulates the C3 sensors (acc, gyro, gps, peripherals, dm state, cameras) to OpenPilot"""

  def __init__(self, dual_camera=False, camera_transport_config=None, camera_telemetry=None):
    self.pm = messaging.PubMaster(['accelerometer', 'gyroscope', 'gpsLocationExternal', 'driverStateV2', 'driverMonitoringState', 'peripheralState',
                                  'deviceMotion'])
    self.camerad = Camerad(dual_camera=dual_camera)
    self.last_perp_update = 0
    self.last_dmon_update = 0
    config = camera_transport_config or {}
    self._fault_delay_ms = int(config.get("target_delay_ms", 0))
    self._camera_frame_ids = {"road": 0, "wide": 0}
    source_frames = os.environ.get("SIMLAB_CAMERA_DEBUG_SOURCE_FRAME_IDS", "")
    self._debug_source_frame_ids = {int(value) for value in source_frames.split(",") if value}
    self._debug_capture_dir = os.environ.get("SIMLAB_CAMERA_DEBUG_SOURCE_DIR")
    if self._debug_source_frame_ids and not self._debug_capture_dir:
      raise ValueError("SIMLAB_CAMERA_DEBUG_SOURCE_DIR is required when camera source frames are configured")
    self.camera_transport = CameraTransportDelay(
      self._publish_camera_frame, camera_telemetry,
      target_delay_ms=0, capacity_frames=int(config.get("queue_capacity_frames", 8)),
    )

  def enable_camera_transport_fault(self, enabled: bool):
    self.camera_transport.set_target_delay_ms(self._fault_delay_ms if enabled else 0)

  def close(self):
    self.camera_transport.close()

  def _publish_camera_frame(self, frame: QueuedCameraFrame):
    if frame.camera == "road":
      self.camerad.cam_send_yuv_road(frame.yuv, frame.source_frame_id, frame.capture_mono_ns)
    else:
      self.camerad.cam_send_yuv_wide_road(frame.yuv, frame.source_frame_id, frame.capture_mono_ns)

  def send_imu_message(self, simulator_state: 'SimulatorState'):
    for _ in range(5):
      dat = messaging.new_message('accelerometer', valid=True)
      dat.accelerometer.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
      dat.accelerometer.init('acceleration')
      dat.accelerometer.acceleration.v = [simulator_state.imu.accelerometer.x, simulator_state.imu.accelerometer.y, simulator_state.imu.accelerometer.z]
      self.pm.send('accelerometer', dat)

      dat = messaging.new_message('gyroscope', valid=True)
      dat.gyroscope.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
      dat.gyroscope.init('gyroUncalibrated')
      dat.gyroscope.gyroUncalibrated.v = [simulator_state.imu.gyroscope.x, simulator_state.imu.gyroscope.y, simulator_state.imu.gyroscope.z]
      self.pm.send('gyroscope', dat)

  def send_gps_message(self, simulator_state: 'SimulatorState'):
    if not simulator_state.valid:
      return

    # transform from vel to NED
    velNED = [
      -simulator_state.velocity.y,
      simulator_state.velocity.x,
      simulator_state.velocity.z,
    ]

    for _ in range(10):
      dat = messaging.new_message('gpsLocationExternal', valid=True)
      dat.gpsLocationExternal = {
        "unixTimestampMillis": int(time.time() * 1000),  # noqa: TID251
        "flags": 1,  # valid fix
        "horizontalAccuracy": 1.0,
        "verticalAccuracy": 1.0,
        "speedAccuracy": 0.1,
        "bearingAccuracyDeg": 0.1,
        # locationd uses this explicit field; flags alone are not interpreted
        # by the Python bridge like they are by the ublox decoder.
        "hasFix": True,
        "satelliteCount": 12,
        "vNED": velNED,
        "bearingDeg": simulator_state.imu.bearing,
        "latitude": simulator_state.gps.latitude,
        "longitude": simulator_state.gps.longitude,
        "altitude": simulator_state.gps.altitude,
        "speed": simulator_state.speed,
        "source": log.GpsLocationData.SensorSource.ublox,
      }

      self.pm.send('gpsLocationExternal', dat)

  def send_device_motion(self, simulator_state: 'SimulatorState'):
    dat = messaging.new_message('deviceMotion', valid=True)
    motion = dat.deviceMotion
    motion.orientationNED = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'valid': True}
    motion.velocityDevice = {'x': simulator_state.velocity.x, 'y': simulator_state.velocity.y,
                             'z': simulator_state.velocity.z, 'valid': True}
    motion.angularVelocityDevice = {'x': simulator_state.imu.gyroscope.x, 'y': simulator_state.imu.gyroscope.y,
                                    'z': simulator_state.imu.gyroscope.z, 'valid': True}
    motion.accelerationDevice = {'x': simulator_state.imu.accelerometer.x, 'y': simulator_state.imu.accelerometer.y,
                                 'z': simulator_state.imu.accelerometer.z, 'valid': True}
    motion.inputsOK = motion.sensorsOK = motion.posenetOK = True
    motion.timestamp = dat.logMonoTime
    self.pm.send('deviceMotion', dat)

  def send_peripheral_state(self):
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    self.pm.send('peripheralState', dat)

  def send_fake_driver_monitoring(self):
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.rightDriverData.faceProb = 1.0
    self.pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState', valid=True)
    dm = dat.driverMonitoringState
    dm.alertLevel = log.DriverMonitoringState.AlertLevel.none
    dm.activePolicy = log.DriverMonitoringState.MonitoringPolicy.vision
    dm.visionPolicyState.faceDetected = True
    dm.visionPolicyState.isDistracted = False
    dm.visionPolicyState.awarenessPercent = 100
    self.pm.send('driverMonitoringState', dat)

  def send_camera_images(self, world: 'World'):
    world.image_lock.acquire()
    # Copy immediately after the producer signal. Conversion and delay scheduling
    # happen after the copy so a transport fault cannot hold simulator memory.
    road_rgb = world.road_image.copy()
    wide_rgb = world.wide_road_image.copy() if world.dual_camera else None
    capture_mono_ns = time.monotonic_ns()

    road_id = self._camera_frame_ids["road"]
    self._camera_frame_ids["road"] += 1
    if road_id in self._debug_source_frame_ids:
      capture_dir = Path(self._debug_capture_dir)
      capture_dir.mkdir(parents=True, exist_ok=True)
      image_path = capture_dir / f"camera-source-frame-{road_id:06d}.png"
      Image.fromarray(road_rgb).save(image_path)
      (capture_dir / f"{image_path.name}.json").write_text(json.dumps({
        "schema_version": 1,
        "scope": "analysis_only_exact_pre_nv12_model_input",
        "camera": "road",
        "source_frame_id": road_id,
        "capture_mono_ns": capture_mono_ns,
      }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    self.camera_transport.enqueue(camera="road", source_frame_id=road_id, capture_mono_ns=capture_mono_ns,
                                  yuv=self.camerad.rgb_to_yuv(road_rgb))
    if wide_rgb is not None:
      wide_id = self._camera_frame_ids["wide"]
      self._camera_frame_ids["wide"] += 1
      self.camera_transport.enqueue(camera="wide", source_frame_id=wide_id, capture_mono_ns=capture_mono_ns,
                                    yuv=self.camerad.rgb_to_yuv(wide_rgb))

  def update(self, simulator_state: 'SimulatorState', world: 'World'):
    now = time.monotonic()
    self.send_imu_message(simulator_state)
    self.send_gps_message(simulator_state)
    self.send_device_motion(simulator_state)

    if (now - self.last_dmon_update) > DT_DMON/2:
      self.send_fake_driver_monitoring()
      self.last_dmon_update = now

    if (now - self.last_perp_update) > 0.25:
      self.send_peripheral_state()
      self.last_perp_update = now
