"""Asynchronous RGB capture for CARLA analysis datasets.

This is deliberately outside the control path: the camera callback only queues
an immutable copy and the writer records an image plus source-frame metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from queue import Full, Queue
import threading
from typing import Any, Callable

import numpy as np


class CaptureWriter:
  def __init__(self, directory: str | None, every_n_frames: int, emit: Callable[[dict[str, Any]], None]):
    if every_n_frames < 1:
      raise ValueError("every_n_frames must be positive")
    self.directory = Path(directory) if directory else None
    self.every_n_frames, self.emit = every_n_frames, emit
    self._queue: Queue[tuple[int, int, np.ndarray] | None] = Queue(maxsize=16)
    self._thread: threading.Thread | None = None
    self._overflow_count = 0
    if self.directory is not None:
      self.directory.mkdir(parents=True, exist_ok=True)
      self._thread = threading.Thread(target=self._write, name="carla-capture-writer", daemon=True)
      self._thread.start()

  def offer(self, source_frame_id: int, capture_mono_ns: int, rgb: np.ndarray) -> None:
    if self.directory is None or source_frame_id % self.every_n_frames:
      return
    try:
      self._queue.put_nowait((source_frame_id, capture_mono_ns, rgb.copy()))
    except Full:
      self._overflow_count += 1
      self.emit({"type": "dataset_capture_drop", "source_frame_id": source_frame_id,
                 "capture_mono_ns": capture_mono_ns, "reason": "writer_queue_overflow"})

  def _write(self) -> None:
    from PIL import Image
    while True:
      item = self._queue.get()
      if item is None:
        return
      source_frame_id, capture_mono_ns, rgb = item
      filename = f"road-frame-{source_frame_id:06d}.png"
      path = self.directory / filename
      Image.fromarray(rgb).save(path)
      metadata = {"type": "dataset_capture", "source_frame_id": source_frame_id,
                  "capture_mono_ns": capture_mono_ns, "image": f"captures/{filename}"}
      path.with_suffix(".json").write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
      self.emit(metadata)

  def close(self) -> int:
    if self._thread is None:
      return 0
    self._queue.put(None)
    self._thread.join(timeout=10)
    if self._thread.is_alive():
      raise RuntimeError("CARLA capture writer did not flush")
    (self.directory / "capture_status.json").write_text(
      json.dumps({"schema_version": 1, "dropped": self._overflow_count}, sort_keys=True) + "\n", encoding="utf-8")
    return self._overflow_count
