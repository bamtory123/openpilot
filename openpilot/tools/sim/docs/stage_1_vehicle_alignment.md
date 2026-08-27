# Stage 1 — simulator and virtual-vehicle alignment

## Status

In progress. The first identified CAN contract defect has been corrected for the `HONDA_CIVIC_2022` simulator profile.

## 2026-08-22: stock-ACC status correction

### Finding

The simulated `POWERTRAIN_DATA.ACC_STATUS` mirrored `selfdriveState.active`. For the Honda Civic 2022 profile, openpilot controls longitudinal behavior directly in the simulator. Reporting a simultaneously enabled stock PCM ACC made `selfdrived` emit repeated `cruiseMismatch` events.

### Change

`openpilot/tools/sim/lib/simulated_car.py` now reports `ACC_STATUS = 0`, representing the absence of stock PCM ACC while openpilot longitudinal control is active. This does not disable a safety check: it supplies the virtual vehicle state that the scenario actually models.

### Verification

CARLA `curve_60s` run `run-001-20260822T133146Z-e3d333d4` reached active state and recorded **zero** `cruiseMismatch` events. It still collided after 15.357 s, so the overall scenario verdict was correctly `FAIL`; that lateral-control result belongs to Stage 2 and is not masked by this CAN correction.

### Remaining Stage 1 check

Re-run the fixed CARLA curve scenario with the new camera/steering contract. Compare its camera manifest, command-to-response trace, and event list with the retained pre-change baseline before beginning controller tuning.

## 2026-08-22: camera and steering contract

### Camera

The CARLA road camera uses the narrow-road resolution of 1928×1208 and a focal length of 2648 px, giving a 40.03° horizontal FOV. The original mount copied from the MetaDrive C3 baseline, `(x=0.0 m, y=0.0 m, z=1.22 m, pitch=0°)`, was later found to lie inside CARLA's Tesla cabin because the vehicle coordinate origins differ. Stage 2 corrects the CARLA-local mount to `(x=1.45 m, y=0.0 m, z=1.35 m, pitch=0°)`; see `stage_2_perception_trajectory_contract.md` for the captured-frame evidence.

### Steering conversion

The bridge now converts `steeringAngleDeg` using the Honda Civic 2022 steer ratio (15.38) and CARLA's reported front-wheel limit rather than an arbitrary 70° divisor. The current CARLA Tesla Model 3 reports a 70.0° maximum front-wheel angle, yielding a 1076.6° steering-wheel command limit.

`openpilot/tools/sim/bridge/carla/steering_calibration.py` measures simulator physics independently of openpilot. The latest straight-road calibration at 8 m/s recorded symmetric yaw response:

| Normalized CARLA steer | Observed yaw rate |
| ---: | ---: |
| -0.05 | -7.7604 °/s |
| -0.02 | -3.1370 °/s |
| +0.02 | +3.1370 °/s |
| +0.05 | +7.7603 °/s |

The raw artifact is stored with the evaluation workspace at `outputs/carla-harness/calibration/steering_response.json`.

## Contract comparison run

The first `curve_60s` run after the camera/steering contract change was `run-001-20260822T133736Z-557e32ac`. It is compared below with the retained pre-change curve baseline `run-001-20260822T132619Z-ea59c80c`.

| Metric | Before | Camera/steering contract | Result |
| --- | ---: | ---: | --- |
| Active time | 23.401 s | 15.291 s | Worse |
| Route progress | 68.75% | 65.62% | Worse |
| Mean lane-centre error | 0.7005 m | 0.6222 m | Better |
| Maximum lane-centre error | 3.5933 m | 3.3822 m | Better |
| Lane invasions | 21 | 17 | Better |
| Collision | 1 | 1 | Not acceptable |
| Mean speed | 4.1457 m/s | 5.7378 m/s | Higher, but unsafe |

The run remains a `FAIL`: it collided before the 60 s target. Keep the new camera/steering contract because it replaces demonstrably inconsistent geometry and unit conversion, but do **not** claim a driving-performance improvement. The next work is Stage 2: diagnose the excessive speed/curve response with fixed-speed and steering-response traces, then tune one lateral or longitudinal behavior at a time.

## 2026-08-22: closed-loop curvature telemetry diagnosis

The CARLA harness now records the whole lateral-control chain on every run:

1. `modelV2.action.desiredCurvature` — learned model target
2. `controlsState.desiredCurvature` — controller/planner target
3. `carControl.actuators.steeringAngleDeg` and the normalized CARLA steer passed by the bridge
4. CARLA transform/velocity-derived yaw rate and actual curvature

On `curve_60s` run `run-001-20260822T135016Z-703ed5a3`, both the model and planner target were effectively zero throughout the bend (`model_k` and `plan_k` displayed as `±0.0000 1/m`). The subsequent signals were consistent with that input: the command stayed at `+0.04°` to `+0.37°`, the bridge sent only `+0.00006` to `+0.00035` normalized steer, and CARLA measured about `+0.0001 1/m` actual curvature.

This rules out steering conversion and vehicle physics as the primary cause of this failure. The immediate Stage 2 task is a **perception/trajectory contract diagnosis**: make CARLA imagery and route geometry recognizable to the learned driving model, or introduce a separately labelled oracle-route controller benchmark. Controller-gain tuning must wait until the target curvature becomes non-zero and plausible for the road geometry.
