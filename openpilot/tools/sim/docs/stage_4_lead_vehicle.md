# Stage 4 — lead vehicle and stop/restart evaluation

## Scope and safety boundary

CARLA lead position, relative speed, and TTC are **evaluator-only ground truth**. They are emitted to the local harness only and are never published into openpilot's perception, planning, or control inputs. Results therefore remain camera-only openpilot evaluations, not oracle-assisted control claims.

## Implemented scenarios

| Scenario | Ego entry speed | Lead behavior |
| --- | ---: | --- |
| `lead_follow_30` | 50 km/h | Same-lane lead begins 35 m ahead at 30 km/h |
| `lead_stop_restart` | 40 km/h | Same-lane lead begins 35 m ahead at 8 m/s, stops 8–12 s after engagement, then resumes |

The lead is a kinematic evaluator actor. It remains stationary through bridge warm-up, starts only after openpilot engagement, and is not fed back as CAN, radar, or any oracle message. This avoids CARLA traffic-manager nondeterminism and preserves the intended initial gap.

Harness metrics include minimum Euclidean lead distance, maximum closing speed, and minimum positive TTC. The active-run watchdog terminates an unresponsive bridge at `drive-seconds + grace-seconds` after engagement, rather than waiting through the startup timeout.

## Initial validation — 2026-08-22

After lead placement was corrected to use the known pre-tick ego spawn transform, `lead_follow_30` run `run-001-20260822T141510Z-df748d19` recorded the intended 35 m starting gap, `14.8876 m` minimum lead distance, `3.3627 m/s` maximum closing speed, and `8.6884 s` minimum TTC. It still failed due to the existing lateral collision after 8.291 active seconds; this must not be interpreted as a valid following-control verdict.

`lead_stop_restart` run `run-001-20260822T141541Z-7947e4c1` verified telemetry through the scenario (`28.3479 m` minimum lead distance and `9.5027 s` minimum TTC), but CARLA ceased responding during shutdown. The harness surfaced `drive_timeout_after_engagement` and `sensorDataInvalid` rather than hanging indefinitely. CARLA must be restarted before the next run.

## Acceptance gate after Stage 2 and 3

1. Follow at least three repetitions with no collision, no unexplained lane departure, and no TTC below the recorded safety threshold.
2. Verify a controlled stop and restart in three repetitions before adding a cut-in scenario.
3. Label any future test that injects CARLA actor data into openpilot as `oracle-assisted`; do not mix it with camera-only results.
