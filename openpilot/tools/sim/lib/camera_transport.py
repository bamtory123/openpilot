"""Non-blocking camera transport-delay support for simulator experiments.

Frames are copied and converted by the producer before being queued. The
publisher owns the delay schedule so a fault never stalls camera capture.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import threading
import time
from collections.abc import Callable


@dataclass(frozen=True, order=True)
class QueuedCameraFrame:
  release_mono_ns: int
  sequence: int
  camera: str
  source_frame_id: int
  capture_mono_ns: int
  yuv: bytes


class CameraTransportDelay:
  """Schedules immutable camera frames without blocking the capture thread."""

  def __init__(self, publisher: Callable[[QueuedCameraFrame], None], telemetry: Callable[[dict], None] | None = None,
               *, target_delay_ms: int = 0, capacity_frames: int = 8,
               clock: Callable[[], int] = time.monotonic_ns):
    if target_delay_ms < 0 or capacity_frames < 1:
      raise ValueError("delay must be non-negative and capacity must be positive")
    self.publisher = publisher
    self.telemetry = telemetry or (lambda _: None)
    self.clock = clock
    self.capacity_frames = capacity_frames
    self.target_delay_ns = target_delay_ms * 1_000_000
    self._queue: list[QueuedCameraFrame] = []
    self._sequence = 0
    self._closed = False
    self._condition = threading.Condition()
    self._thread = threading.Thread(target=self._run, name="camera-delay-publisher", daemon=True)
    self._thread.start()

  def set_target_delay_ms(self, delay_ms: int) -> None:
    if delay_ms < 0:
      raise ValueError("delay must be non-negative")
    with self._condition:
      self.target_delay_ns = delay_ms * 1_000_000
      self._condition.notify_all()

  def enqueue(self, *, camera: str, source_frame_id: int, capture_mono_ns: int, yuv: bytes) -> bool:
    """Return false on overflow; the producer never waits for release time."""
    with self._condition:
      if self._closed:
        return False
      if len(self._queue) >= self.capacity_frames:
        self.telemetry({
          "type": "camera_frame", "camera": camera, "source_frame_id": source_frame_id,
          "capture_mono_ns": capture_mono_ns, "dropped": True, "drop_reason": "queue_overflow",
          "queue_depth": len(self._queue),
        })
        return False
      frame = QueuedCameraFrame(
        release_mono_ns=capture_mono_ns + self.target_delay_ns,
        sequence=self._sequence,
        camera=camera,
        source_frame_id=source_frame_id,
        capture_mono_ns=capture_mono_ns,
        yuv=yuv,
      )
      self._sequence += 1
      heapq.heappush(self._queue, frame)
      self._condition.notify_all()
      return True

  def close(self) -> None:
    with self._condition:
      self._closed = True
      self._queue.clear()
      self._condition.notify_all()
    self._thread.join(timeout=2)

  def _run(self) -> None:
    while True:
      with self._condition:
        while not self._closed and not self._queue:
          self._condition.wait()
        if self._closed:
          return
        frame = self._queue[0]
        wait_ns = frame.release_mono_ns - self.clock()
        if wait_ns > 0:
          self._condition.wait(wait_ns / 1_000_000_000)
          continue
        frame = heapq.heappop(self._queue)
        queue_depth = len(self._queue)

      self.publisher(frame)
      publish_mono_ns = self.clock()
      self.telemetry({
        "type": "camera_frame", "camera": frame.camera, "source_frame_id": frame.source_frame_id,
        "capture_mono_ns": frame.capture_mono_ns, "scheduled_publish_mono_ns": frame.release_mono_ns,
        "actual_publish_mono_ns": publish_mono_ns,
        "target_delay_ms": (frame.release_mono_ns - frame.capture_mono_ns) / 1_000_000,
        "actual_delay_ms": (publish_mono_ns - frame.capture_mono_ns) / 1_000_000,
        "queue_depth": queue_depth, "dropped": False,
      })
