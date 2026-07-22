# Architecture

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22 alongside the rest of `docs/llm/`. This page describes the
architecture as inherited from `stacking-planner`, flags what still applies to
a manipulator-based tabletop system, and what does not. See
[Overview](overview.md) for the project-level summary of the re-implementation.

## Inherited Stack-Planning Loop

The planning core is expected to stay conceptually the same:

1. Load [OmegaConf](glossary.md#omegaconf) config from `agent/configs/*.yml`.
2. Build the stacking environment and inventory from stone assets.
3. Generate candidate [actions](glossary.md#action) from planar sampling, rotations or scanned poses, pose optimization, and support filters.
4. Use [MCTS](glossary.md#mcts) to select stacking actions.
5. Use `planning/` helpers and `diffsimpy.planner` for [grasp](glossary.md#grasp), IK, and motion planning.
6. Execute or preview the resulting sequence through a desktop GUI/control workflow.
7. Use perception to scan, identify, and resume from the real scene.

Step 6 (execution/control) and step 7 (perception) are the parts that change
substantially for this project; steps 1-5 (the planning core) are expected to
need parameter/scale retuning rather than a structural rewrite, since they are
built around `diffsimpy` physics and are largely end-effector/robot agnostic
at the API level.

## Core Modules (status)

- `agent/integrated_planner.py`: environment/policy/[MCTS](glossary.md#mcts) plumbing and high-level planning/perception-assisted methods. Expected to stay, but its perception-assisted methods (`get_inhand_pose`, scan hooks) are tied to the old NUC-based perception flow and will need new call sites.
- `agent/env/env.py`, `agent/env/simulator.py`: environment/physics wrapper. Expected to stay, subject to scale/tuning changes for tabletop stones.
- `agent/env/components/`: candidate action generation, pose solving, support reasoning, inventory, state. Expected to stay.
- `agent/mcts/`: tree search, nodes, CEM proposal logic. Expected to stay.
- `planning/planning.py`: `diffsimpy.planner.Context` setup and motion-planning helpers. Expected to stay in shape, but grasp/IK bindings assume the excavator gripper model (`SURFACE_GRASP_*` constants, gripper URDF) and will need a manipulator end-effector equivalent.
- `planning/execution*.py`, `gui/execution_window.py`, `scripts/desktop/execute_*.py`: excavator-specific execution workflow (ROS2 phase choreography, NUC-mediated scan handoff, multi-process GUI/worker split). **To be redesigned** for a single-desktop, manipulator-driven control loop.
- `perception/`: point-cloud/mesh/DSF reconstruction utilities are plausibly reusable if the new perception stack still produces per-stone point clouds; `perception/sceneid_runtime.py` and the LiDAR/ICP-based scene identification flow are excavator-specific and **to be redesigned**, likely replaced or supplemented by AprilTag-based pose identification (undecided).
- `ros2/`: node wrappers modeled around excavator joints/phases and NUC-relayed topics. **To be redesigned** for the manipulator's control interface.

## Data Contracts (inherited, likely still valid)

These are conventions used throughout the planning/observation code and are
not obviously excavator-specific; they should carry over unless the
manipulator/perception redesign has a reason to change them:

- Stone poses are usually 7-vectors: `[x, y, z, qx, qy, qz, qw]`.
- Quaternion order is `(x, y, z, w)`, matching SciPy/Open3D usage.
- Face padding uses `-1`; vertex padding uses `np.inf`. Do not change casually; consumed by graph/edge builders.
- Height-map observations include x/y bounds as part of the observation/training contract.
- MCTS h5 rows include state/action/reward/next-state data, sampled actions, MCTS Q/policy values, done/failed flags, visits, and step metadata.

## External Boundary

- [`diffsimpy`](glossary.md#diffsimpy) is provided by the sibling `../diffsim` repo; expected to remain a shared dependency with `stacking-planner`.
- Planner, [posegen](glossary.md#posegen), [poseinit](glossary.md#poseinit), and diffsim binding mismatches are often cross-repo issues rather than pure Python bugs in this repository.

## Removed From This Page (2026-07-22 reset)

The excavator-specific perception/resume flow (NUC-mediated scene scans over
Wi-Fi, SceneID reconstruction from logs) previously documented here has been
removed as not applicable to a desktop-only, likely-AprilTag-based system.
See `../stacking-planner/docs/llm/architecture.md` for that historical
content if it is useful reference while designing the new perception flow.

## TODO

- Draw the runtime process topology once the manipulator control interface (ROS2 driver or otherwise) is decided.
- Decide whether the GUI/worker multi-process split (used to keep excavator ROS2/Ray/diffsim calls off the GUI thread) is still needed for a manipulator control loop, or can be simplified.
- Document the manipulator/end-effector kinematic model once assets exist under `model/`/`assets/`.
