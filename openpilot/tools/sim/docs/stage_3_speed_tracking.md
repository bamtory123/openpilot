# Stage 3 — speed tracking and longitudinal comfort

## Scope and safety boundary

These scenarios evaluate the CARLA/openpilot integration; they do not alter openpilot's longitudinal controller, cruise safety events, or actuator limits. A scenario entry speed is evaluator configuration, not an oracle signal supplied to openpilot.

## Implemented scenarios

| Scenario | Entry speed | Intended probe |
| --- | ---: | --- |
| `straight_30` | 30 km/h | Low-speed straight hold |
| `straight_50` | 50 km/h | Medium-speed straight hold |
| `straight_70` | 70 km/h | Higher-speed straight hold |
| `curve_60s` | 28.8 km/h | Curve approach and response |

The CARLA bridge records the configured entry speed as evaluator telemetry. The harness calculates mean/max absolute speed error, acceleration, and jerk from CARLA/CAN state and preserves the raw samples for every run.

## Initial baseline — 2026-08-22

`straight_30` run `run-001-20260822T140451Z-6de724b1` recorded a mean speed of `8.6976 m/s` against a `8.3333 m/s` entry target, but failed after 10.816 active seconds with one collision and 17 lane invasions. Mean speed error was `1.5176 m/s`, maximum acceleration `5.2564 m/s²`, and maximum sampled jerk `1314.1109 m/s³`.

This baseline is not a longitudinal-controller calibration result: the simultaneous lateral/perception instability invalidates comfort claims. Stage 3 infrastructure is complete, but acceptance remains blocked until Stage 2 produces collision-free lane tracking.

## Acceptance gate after Stage 2

Run each fixed-speed scenario at least three times. Require zero collision and lane departure, stable active duration, and recorded bounds for speed error, acceleration, and jerk before accepting an acceleration-to-throttle/brake mapping change. CARLA-specific translation changes must remain isolated from the MetaDrive bridge.
