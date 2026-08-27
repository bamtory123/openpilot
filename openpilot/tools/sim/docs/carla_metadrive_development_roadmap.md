# openpilot simulator development roadmap

## Purpose and safety boundary

This workspace is for reproducible **simulation-only** experiments with the `HONDA_CIVIC_2022` virtual CAN profile. A simulator pass does not establish real-vehicle safety or authorize a real-vehicle change. Do not weaken openpilot safety events, no-entry gates, or actuator limits to make an evaluation pass.

The current CARLA bridge is a working integration baseline: Windows-native CARLA provides RGB frames; the Ubuntu WSL checkout runs openpilot; and the evaluator records active-drive, route, lane, and collision results. MetaDrive remains the fast smoke/regression backend.

## Stage 1 — simulator and virtual-vehicle alignment

### Goal

Make CARLA state, the virtual Honda CAN messages, and the openpilot device model describe the same vehicle consistently before changing driving behavior.

### Work items

1. Calibrate the CARLA road-camera mount, height, pitch, FOV, and image resolution against the simulated C3 camera contract.
2. Replace the current approximate steering-wheel-degrees to CARLA steering mapping with a measured, bounded mapping; report the applied steering command and physical vehicle response separately.
3. Complete the Honda cruise/CAN feedback used by `HONDA_CIVIC_2022` so `cruiseMismatch` is not a normal outcome of a valid simulator run.
4. Keep GPS/CAN velocity authoritative until a CARLA-to-device IMU rigid transform has been calibrated. Do not feed Unreal-world acceleration directly as device-frame IMU data.

### Exit criteria

- No persistent `cruiseMismatch` while active.
- Camera and CAN contracts are captured in the run manifest.
- A fixed straight scenario has stable speed feedback and bounded steering response.

## Stage 2 — lane keeping and curve tracking

### Goal

Evaluate lateral control on fixed, repeatable routes before comparing tuning changes.

### Work items

1. Use `curve_60s` in Town04: deterministic curved route, 60 s target duration, controlled entry speed, route-progress gate.
2. Record route length/curvature, route progress, mean and maximum lane-centre error, lane invasions, collision count, steering angle, and steering rate.
3. Change one lateral parameter or simulator mapping at a time; retain the previous baseline and compare identical runs.

### Initial acceptance targets

- Collision count: 0.
- Lane invasions: 0 or explicitly justified.
- Mean lane-centre error: under 0.3 m.
- Maximum lane-centre error: under 0.8 m.
- Complete the configured duration and route-progress gate.

## Stage 3 — speed tracking and longitudinal comfort

### Goal

Make the virtual Honda cruise state, openpilot longitudinal plan, and CARLA throttle/brake dynamics agree on an appropriate speed.

### Work items

1. Add fixed straight speed targets (30, 50, and 70 km/h) and a curve-speed scenario.
2. Calibrate CARLA-specific acceleration-to-throttle/brake translation without changing the shared MetaDrive mapping.
3. Evaluate speed error, settling time, maximum acceleration, jerk, and stop/restart behavior.

### Exit criteria

- Target-speed tracking is stable without cruise-state errors.
- Jerk and acceleration remain within explicitly recorded thresholds.
- Curves induce safe deceleration rather than lane departure or collision.

## Stage 4 — lead vehicle and obstacle scenarios

### Goal

Evaluate longitudinal response to a lead vehicle without confusing evaluator ground truth with openpilot perception.

### Work items

1. Add a deterministic same-lane lead vehicle at fixed initial distance and relative speed.
2. Record CARLA ground-truth lead distance, relative speed, collision, and minimum TTC **for evaluation only**.
3. First test following, braking, stopping, and restarting. Add cut-ins only after those pass.
4. Label any experiment that feeds CARLA ground truth into openpilot as an oracle-assisted experiment, not a camera-only result.

### Scope limit

openpilot is an L2 driver-assistance stack, not a general obstacle-navigation planner. It should not be represented as a system that autonomously chooses an arbitrary lane-change path around obstacles. General obstacle avoidance requires a separate high-level behavior/path-planning project and remains simulation-only unless independently validated.

## Experiment discipline

For each change, retain the source revision, scenario parameters, environment, result JSON, and before/after comparison. Start with one diagnostic run; accept a tuning change only after the configured multi-run regression gate passes.

## Current implementation status — 2026-08-22

Stage 3 and Stage 4 scenario infrastructure is implemented and documented in `stage_3_speed_tracking.md` and `stage_4_lead_vehicle.md`. Their stock-openpilot baselines are intentionally recorded as failures because Stage 2 lane/perception tracking is still failing. Do not tune longitudinal gains or claim lead-following capability until the Stage 2 safety gate passes.
