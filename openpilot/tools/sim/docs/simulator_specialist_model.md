# CARLA simulator-specialist model path

## Purpose and boundary

This is a separate, simulation-only visual model path for CARLA. It does not replace the stock openpilot model and it must not be used for a real vehicle. The first implementation is a dependency-free ridge regressor over downsampled RGB road-view features; it establishes the data, label, artifact, and shadow-evaluation contracts needed for a later neural model.

At inference the specialist is **shadow-only**: it publishes an evaluator telemetry value and never writes `modelV2`, `controlsState`, CAN, or vehicle controls.

## Data contract

`tools/sim/carla_specialist/collect.py` creates:

- `images/*.png`: 1928×1208 RGB frames from the CARLA windshield camera.
- `labels.jsonl`: map-waypoint future curvature in openpilot sign convention.
- `manifest.json`: town, seed, camera contract, and sample count.

The map curvature is a training label during offline dataset generation. It is not available to the model at inference and it is never fed into stock openpilot.

## Workflow

From the openpilot checkout in WSL, after CARLA is reachable:

```bash
SIMULATION=1 PYTHONPATH=/home/hyunsung/src/openpilot \
  .venv/bin/python openpilot/tools/sim/carla_specialist/collect.py \
  --host 172.28.112.1 --town Town04 --samples 200 \
  --output /mnt/c/Users/Hyunsung\ Kim/Documents/Codex/2026-08-22/op/outputs/carla-specialist/dataset-town04

SIMULATION=1 PYTHONPATH=/home/hyunsung/src/openpilot \
  .venv/bin/python openpilot/tools/sim/carla_specialist/train.py \
  /mnt/c/Users/Hyunsung\ Kim/Documents/Codex/2026-08-22/op/outputs/carla-specialist/dataset-town04 \
  --output /mnt/c/Users/Hyunsung\ Kim/Documents/Codex/2026-08-22/op/outputs/carla-specialist/town04-ridge.npz
```

Use the artifact in a shadow benchmark:

```bash
SIMULATION=1 PYTHONPATH=/home/hyunsung/src/openpilot \
  .venv/bin/python /mnt/c/Users/Hyunsung\ Kim/Documents/Codex/2026-08-22/op/outputs/carla-harness/carla_openpilot_harness.py \
  --carla-host 172.28.112.1 --town Town04 --scenario curve_60s --runs 1 \
  --drive-seconds 60 --specialist-model /mnt/c/Users/Hyunsung\ Kim/Documents/Codex/2026-08-22/op/outputs/carla-specialist/town04-ridge.npz
```

The harness records `mean_abs_specialist_to_route_curvature_error_1pm` and `specialist_route_curvature_direction_match_fraction` next to stock-model metrics.

## Perception monitoring

Use `--capture-camera-frames` with a harness run, then render its report:

```bash
SIMULATION=1 PYTHONPATH=/home/hyunsung/src/openpilot \
  .venv/bin/python openpilot/tools/sim/carla_specialist/render_report.py RUN_DIRECTORY
```

The resulting `perception-report.png` overlays qualitative future paths on the exact RGB frames supplied to the model: green is CARLA map reference, blue stock model, yellow specialist, and red actual CARLA vehicle curvature. This makes three failures distinguishable: visual prediction disagrees with the road, specialist disagrees with stock, or the physical vehicle fails to follow the requested path. The `run_carla_specialist_assist.ps1` launcher captures frames and opens this report automatically.

## Promotion rules

1. Hold out route locations and lighting conditions from training.
2. Require a better curvature error and direction match than stock openpilot in three shadow repetitions.
3. Only then consider a separately labelled simulator-assist experiment. It must retain stock controls as a baseline and must never be called a stock-openpilot or real-vehicle result.

## Initial smoke artifact — 2026-08-22

The first Town04 collection was interrupted when CARLA became unresponsive after 28 valid frames, so it is intentionally a pipeline smoke dataset rather than a training-quality corpus. The resulting `town04-smoke-ridge.npz` artifact trained on 22 frames and held out 6; its validation MAE was `0.001372 1/m` and direction match was `50%`. It is suitable only to validate loading and shadow telemetry. It must not be used to steer the simulator.

Before any performance comparison, recollect at least 200 frames across held-out locations and lighting conditions after CARLA server stability has been fixed.

## Shadow integration check — 2026-08-22

`curve_60s` run `run-001-20260822T142448Z-a20ae112` loaded the smoke artifact with `carla_specialist_mode: shadow`. It remained a stock-control run and collided after 10.03 active seconds. On the curved samples, stock model-to-route error was `0.005701 1/m` with 81.26% direction match; specialist-to-route error was `0.005023 1/m` with 78.83% direction match. The specialist's magnitude error is marginally lower, but its directional consistency is worse and the sample is far too small. The artifact remains shadow-only.

## Explicit simulator-assist check — 2026-08-22

`run-001-20260822T142733Z-a8665e3f` used `--specialist-mode assist`, which replaces only CARLA's lateral steer with the specialist prediction after engagement. Its run manifest records `carla_specialist_mode: assist` and sample telemetry records `carla_control_source: sim_specialist_assist`; stock openpilot was not modified. The smoke model still collided after 10.807 active seconds, so this is a visibly different simulator-control experiment, not an improvement claim. Assist output is bounded to ±0.03 normalized CARLA steer, low-pass filtered, and now gated to active driving only.

## Path-CNN data-quality gate — 2026-08-23

The collector now records five future centre-path points (5, 10, 20, 30, and 40 m), lane-centre aliases, weather, and route-start coordinates. `--min-abs-curvature 0.002` produces curve-focused samples, preventing a mostly straight-road dataset from inflating an average error.

Two TorchScript path-CNN experiments were run with ClearSunset held out from training. The broad 200-frame Town04 set gave a held-out curvature MAE of `0.002579 1/m`, but its direction metric was dominated by zero-curvature samples and is not a promotion result. The 200-frame curve-focused Town04 set exposed the actual generalisation gap: held-out path lateral MAE `4.3813 m`, curvature MAE `0.028095 1/m`, and direction match `44%`. The model is therefore **rejected** and is not loaded by the CARLA bridge.

The next collection must use multiple towns, independently held-out route-start groups, and substantially more curved examples before retraining. Keep the current production boundary: only an artifact that beats stock openpilot across three live CARLA shadow runs may be considered for a separate simulator-assist experiment.
