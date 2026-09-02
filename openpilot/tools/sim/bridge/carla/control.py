"""Pure CARLA control conversion used by the adapter and contract tests."""

from __future__ import annotations


def normalized_steer(steering_wheel_deg: float, wheel_limit_deg: float) -> float:
  if wheel_limit_deg <= 0:
    raise ValueError("wheel_limit_deg must be positive")
  return max(-1.0, min(1.0, steering_wheel_deg / wheel_limit_deg))
