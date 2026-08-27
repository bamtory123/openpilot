# Stage 2 — CARLA perception and trajectory contract

## Scope

This stage determines whether stock openpilot produces a plausible curvature target from the CARLA camera stream. It deliberately does **not** tune lateral-controller gains. A zero or incorrect model target cannot be repaired by changing steering gains.

## Ground-truth comparison

The `curve_60s` bridge publishes `carla_route_reference_curvature_1pm`: signed centre-line curvature computed from CARLA's map waypoints 4 m behind to 24 m ahead of the ego and converted into openpilot's lateral-sign convention. It is evaluator telemetry only; it is never sent to openpilot and never affects steering, throttle, or braking.

Each harness sample now compares:

| Signal | Source | Purpose |
| --- | --- | --- |
| `carla_route_reference_curvature_1pm` | CARLA road graph | Independent road-shape reference |
| `model_desired_curvature_1pm` | `modelV2` | Learned visual driving target |
| `controls_desired_curvature_1pm` | `controlsState` | Target actually used by controls |
| `commanded_steering_angle_deg` | `carControl` | openpilot output |
| `carla_actual_curvature_1pm` | CARLA pose and velocity | Vehicle response |

The harness summary includes reference/model magnitude error and direction-match fraction only on non-straight route samples (absolute road curvature at least `0.001 1/m`).

## Baseline finding — 2026-08-22

Run `run-001-20260822T135344Z-62687174` entered a section whose CARLA reference curvature rose from about `+0.0019` to `+0.0068 1/m`. At the same timestamps, both openpilot targets displayed as approximately `0.0000 1/m`; the largest observed command was only `+0.42°`, and actual CARLA curvature remained about `+0.0001 1/m`. The run collided after 17.71 active seconds.

This is a perception/trajectory-contract failure, not evidence for a steering-ratio or CARLA-physics defect.

## Camera-frame review — 2026-08-22

The harness can be invoked with `--capture-camera-frames` to save the actual RGB frames sent to camerad under the run directory. Reviewing `active-01-005.1s.png` from `run-001-20260822T135558Z-65986093` found that the previous CARLA mount `(x=0.0 m, z=1.22 m)` was inside the Tesla cabin: its dashboard and roof occupied a large fraction of the view. The earlier mount copied MetaDrive coordinates without accounting for CARLA's different vehicle origin.

The CARLA mount is now `(x=1.45 m, y=0.0 m, z=1.35 m, pitch=0°)`, keeping the C3-like narrow-road optics while positioning the sensor at the CARLA windshield. Capture run `run-001-20260822T135641Z-ad305055` verified unobstructed road visibility. It also showed non-zero model curvature (up to roughly `-0.0013 1/m` in openpilot convention), but this remains much smaller than the approximately `-0.0068 1/m` route reference. The next comparison run uses the now-normalized curvature sign and retains the focus on magnitude before any controller change.

## Windshield-mount result — 2026-08-22

With the curvature sign normalized, run `run-001-20260822T135747Z-63e56dde` measured a mean absolute CARLA route reference of `0.005962 1/m` and mean absolute model target of `0.000296 1/m`. The mean model-to-route error was `0.005681 1/m`; direction agreed on 79.95% of curved-road samples. This is a material improvement over the obstructed-cabin input: the learned model now usually detects the bend direction, but predicts only about 5% of the required curvature magnitude.

The run remains `FAIL` (collision after 10.41 active seconds). Its 17.12° peak command is a downstream response to an insufficient and unstable trajectory target, so lateral controller gain changes remain out of scope. The next experiment is a single-variable camera extrinsic sweep, beginning with pitch while retaining this windshield position and the same road-reference metrics.

## Pitch sweep — 2026-08-22

The harness now accepts `--camera-pitch-deg`, an evaluator-only CARLA camera override. The default remains `0°`; the override is recorded in each run manifest. One-run screening results are below. All runs use the same Town04 route, windshield mount, focal length, traffic count, and starting condition.

| Pitch | Run | Model magnitude (1/m) | Route magnitude (1/m) | Mean error (1/m) | Direction match | Peak steer | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| +5° | `run-001-20260822T140046Z-9921703b` | 0.000022 | 0.005757 | 0.005735 | 91.93% | 1.35° | Under-reacts; collision |
| 0° | `run-001-20260822T135747Z-63e56dde` | 0.000296 | 0.005962 | 0.005681 | 79.95% | 17.12° | Under-reacts; collision |
| -1° | `run-001-20260822T140150Z-17094220` | 0.000222 | 0.006063 | 0.005934 | 76.28% | 12.72° | Under-reacts; collision |
| -2° | `run-001-20260822T140116Z-389fd1ae` | 0.005511 | 0.006141 | 0.008772 | 71.43% | 120.56° | Unstable; collision/excessive actuation |
| -5° | `run-001-20260822T135918Z-f1405ce0` | 0.034238 | 0.006492 | 0.035220 | 71.04% | 499.92° | Gross over-reaction/excessive actuation |

No pitch is promoted from this screen: the candidates are either severely under-responsive or unstable, and one run per value is insufficient to distinguish camera sensitivity from run-to-run model variability. Keep the default `0°` mount for reproducibility. Any future candidate needs three repetitions and must improve both curvature error and safety (no collision, no excessive-actuation event) before becoming the default.

The next high-value variable is visual domain alignment (road-marking style, lighting/weather, and CARLA rendering) rather than controller gains or further large camera-angle changes.

## Next controlled experiments

1. Retain the fixed image contract (1928×1208, RGB-to-NV12, 20 Hz) and record reference-versus-model curvature for every change.
2. Capture and review CARLA camera frames at engagement and at the onset of curvature; check road visibility, horizon placement, lane-marking contrast, and weather/rendering consistency.
3. Vary one camera extrinsic at a time (height, pitch, then FOV), recording the same metrics. Do not modify controller gains during these trials.
4. If no camera configuration produces non-zero, directionally correct targets, treat this as a simulator-to-real-world visual domain gap. Create a separately labelled oracle-route vehicle-dynamics benchmark for controller/physics work, and keep it out of stock-openpilot performance claims.
5. Only after the model target follows the reference curve should Stage 3 tune steering response and speed planning.
