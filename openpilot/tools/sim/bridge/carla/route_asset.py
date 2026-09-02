"""Validated, immutable CARLA route assets used by the adapter pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class RouteAsset:
  town: str
  points: tuple[tuple[float, float, float, float, float, float], ...]
  turn_plan: tuple[str, ...]
  sha256: str


def load_route_asset(path: Path, *, expected_town: str) -> RouteAsset:
  raw = path.read_bytes()
  data = json.loads(raw)
  if data.get("town") != expected_town:
    raise RuntimeError(f"route asset town mismatch: {data.get('town')} != {expected_town}")
  points = data.get("points")
  if not isinstance(points, list) or len(points) < 80:
    raise RuntimeError("route asset must contain at least 80 points")
  normalized = []
  for point in points:
    if not isinstance(point, list) or len(point) != 6 or not all(isinstance(value, (int, float)) for value in point):
      raise RuntimeError("route asset points must be six numeric transform values")
    normalized.append(tuple(float(value) for value in point))
  turn_plan = data.get("turn_plan", [])
  if not isinstance(turn_plan, list) or any(turn not in {"straight", "left", "right", "u_turn"} for turn in turn_plan):
    raise RuntimeError("route asset has invalid turn_plan")
  return RouteAsset(expected_town, tuple(normalized), tuple(turn_plan), hashlib.sha256(raw).hexdigest())
