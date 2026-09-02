from __future__ import annotations

import math


def narrow_road_fov_deg(width_px: int, focal_length_px: float = 2648.0) -> float:
  return math.degrees(2.0 * math.atan(width_px / (2.0 * focal_length_px)))
