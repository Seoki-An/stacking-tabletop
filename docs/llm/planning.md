# Planning

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22 alongside the rest of `docs/llm/`. Unlike
[Execution](execution.md) and [Perception](perception.md), the MCTS/diffsim
planning core documented here is expected to be largely hardware-agnostic
and reusable for the manipulator-based system, so this page keeps the
algorithm-level content from `stacking-planner` rather than starting empty.
Excavator-specific operational detail (script CLI flags tied to the current
`scripts/desktop/generate_sequence.py` entrypoint, manipulator-distance scoring,
session resume/branch mechanics) has been trimmed; see
`../stacking-planner/docs/llm/planning.md` for that full detail if useful
while re-deriving equivalents for this project.

## Status

`scripts/desktop/generate_sequence.py` is the manipulator-era entrypoint name;
it and its CLI surface will likely be renamed/rewritten as part of the
manipulator re-implementation. The algorithm described below (MCTS search
loop, simulation terminology, candidate validation) is the part expected to
carry over with parameter/scale retuning rather than a structural rewrite.

## Core Planning Loop

1. Load OmegaConf config from `agent/configs/*.yml`.
2. Build `IntegratedPlanner`, environment, and stone inventory.
3. Show an initial preview (scene, target wall, regrasp candidates).
4. For each stacking step:
   - Read current environment state.
   - Call `IntegratedPlanner.plan_one_step(...)`, which builds an observation, creates an MCTS root node, and runs `MonteCarloTreeSearch.search(...)`.
   - Root expansion uses CEM (`_expand_root_with_cem(...)`); non-root expansion calls `env.get_action_samples(...)`.
   - Candidate actions are scored by heuristic priors (or a Q-function, if used).
   - Final validation re-simulates ranked root children and returns feasible root candidates.
   - Try motion planning for MCTS candidates (grasp/IK/regrasp via `planning/planning.py`).
   - On success, commit the step and checkpoint session artifacts.

### Simulation Terminology (keep using these terms consistently)

- **Settling simulation**: `Simulator._simulate_settle()` freezes the existing scene and lets only the candidate stone move. This is pose refinement, not a stability test; its displacement/path are diagnostics only.
- **Short simulation**: `Simulator._simulate()` with `environment.sim.short` limits. Visited MCTS tree nodes use it for velocity integrals, trajectories, rewards, and failure attribution.
- **Long simulation**: the same dynamics pass with `environment.sim.long` limits. Ranked root candidates and promoted descendant paths use it during final validation.
- **Final validation**: the candidate-selection procedure around long simulation plus geometric/support checks — not a separate simulation type.

Candidate actions preserve the direct posegen output as `solved_pose`;
settling updates `pose` to the evaluated placement, and debug records expose
that as `settled_pose`.

### MCTS/CEM Notes

- CEM only samples, mutates, scores, and deduplicates root proposals; it runs no physics itself. A root proposal becomes a depth-1 child and is long-simulated (as a feasibility gate) before it can be expanded further.
- `algorithm.mcts.max_depth` counts node depth with the root at 0 (`max_depth: 1` means root children only).
- Tree preservation (`search(..., preserve_tree=True)`, the default) reuses validated states/subtree structure across steps; promoted children are revalidated via `revalidate_preserved_root(...)` before continuing search.
- See `docs/config_manual_mcts_action.md` for the full MCTS/environment-action config key reference — that file is not manipulator-specific and was left as-is in this reset.

## Candidate Filtering Before Motion

- Posegen samples whose `c_gap` penetration residual exceeds `environment.action.posegen_gap_threshold` are rejected.
- Final validation rejects target-vs-scene and target-vs-place-plane penetration deeper than the negative support contact-gap tolerance.
- Rejection reasons include `target_motion_limit`, `scene_motion_limit`, `nonfinite_simulation`, `place_robustness`, `simulation_unsettled`.

## What Is Excavator-Specific (removed from this page)

The following pieces of the inherited planning code are tied to the
manipulator project and are **not** described here; treat them as needing a
fresh design decision rather than assuming a default:

- `regrasp_xy_pos`/regrasp candidate motion generation tied to the manipulator's reach/regrasp geometry.
- The `excavator_distance` planar scoring term (`planar.score.excavator_distance_axis`, `excavator_xy`) — biases candidate scoring by distance from the manipulator base; not meaningful for a manipulator workspace without redefinition.
- Session resume/branch CLI mechanics (`--resume`, `--resume-state-pkl`, `--branch-start-step`) tied to the current script's file layout; likely to carry over conceptually but the entrypoint itself will change.
- SceneID-driven resume (`--simulate-sceneid-stones`, ground-height-from-SceneID plumbing) — depends on the perception redesign in [Perception](perception.md).

## Related Files

- `agent/integrated_planner.py`
- `agent/mcts/mcts.py`, `agent/mcts/cem.py`
- `planning/planning.py`
- `docs/config_manual_mcts_action.md`
- `docs/llm/execution.md`
- `docs/llm/perception.md`

## TODO

- Confirm which config keys need retuning for tabletop scale once stone assets and target geometry exist.
- Design the manipulator-appropriate replacement for regrasp/manipulator-distance scoring.
- Decide the new entrypoint name/CLI surface (replacing `scripts/desktop/generate_sequence.py`).
