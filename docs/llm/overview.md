# Overview

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

## Wiki Reset (2026-07-22)

This wiki was reset on 2026-07-22. `stacking-tabletop` starts as a full copy of
`../stacking-planner` (an manipulator/gripper stacking planner), and is being
re-implemented into a **manipulator-based tabletop stacking system**. Almost
all prior hardware, communication, and perception documentation described the
manipulator system and no longer applies; it has been cleared from this wiki
rather than carried forward as stale guidance.

`../stacking-planner/docs/llm/` and `../stacking-planner/docs/context.md`
still describe the manipulator system in full detail and remain a useful
reference while re-implementing equivalent behavior here. Treat them as
historical/reference material, not as current fact for this repository.

## Wiki Index

- [Architecture](architecture.md): system structure, what is inherited as-is,
  and what is being redesigned.
- [Planning](planning.md): MCTS/diffsim stacking planning (mostly hardware-agnostic).
- [Execution](execution.md): manipulator execution workflow (to be designed).
- [Perception](perception.md): scene/pose identification (to be redesigned, AprilTag under consideration).
- [Build and Run](build_and_run.md): environment, commands, configs.
- [Debugging](debugging.md): known gotchas and investigation workflows.
- [Decisions](decisions.md): stable design decisions for this project (starts empty after reset).
- [Experiments](experiments.md): experiment results for this project (starts empty after reset).
- [Glossary](glossary.md): project vocabulary and keyword definitions.

## What This Project Is

Stacking-Tabletop re-implements the manipulator-based stone-stacking planner as
a tabletop system driven by a robot manipulator (likely a UR5e). Key intended
changes from `stacking-planner`:

- **Robot**: manipulator + bucket/gripper -> manipulator arm (UR5e candidate) with an end effector TBD.
- **Compute topology**: desktop + onboard NUC over Wi-Fi -> **desktop only**, talking directly to the manipulator and cameras. There is no NUC leg in this project.
- **Scene/pose identification**: LiDAR sweep + ICP/SceneID reconstruction -> likely camera-based, possibly using **AprilTag** fiducials for stone/scene pose identification. This is explicitly still open and to be decided/validated, not yet implemented.
- **Scale**: tabletop-scale stones/workspace instead of field-scale stones, which likely affects sensor choice, motion planning limits, and physical tuning constants (these are not yet re-derived).

What is likely still reusable largely as-is (algorithmic core, hardware-agnostic):

- `agent/` MCTS search, CEM root proposal, environment/state/action components.
- `planning/planning.py` diffsim planner setup for grasp/IK/motion (end-effector-specific bindings will need revisiting for a parallel-jaw/manipulator gripper instead of the manipulator gripper).
- `perception/reconstruction_*` stone-model reconstruction pipeline (mesh/DSF fitting from point clouds), if the new perception stack still produces point clouds per stone.
- `loader/`, `trainer/`-adjacent training scripts for BC-MCTS / height-map models, to the extent training data generation stays hardware-agnostic.

What needs to be redesigned (not yet started):

- `ros2/` node wrappers: currently modeled around manipulator joints/phases and NUC-relayed topics; needs a manipulator-appropriate ROS2 (or other control) interface.
- `planning/execution*.py`, `gui/execution_window.py`, `scripts/desktop/execute_*.py`: manipulator-specific execution workflow (pick/place staging, in-hand scan, NUC-mediated scene scan) needs a manipulator-appropriate rewrite.
- `scripts/nuc/*`: entirely NUC-specific; not applicable to a desktop-only topology. Logic may be ported into desktop-side perception modules instead.
- Scene/pose identification pipeline: SceneID/ICP-from-LiDAR needs to be replaced or heavily adapted depending on the AprilTag/camera decision.

## Top-Level Layout (inherited, subject to change)

- `agent/`: stacking environment, [MCTS](glossary.md#mcts), sampler, integrated planner, configs.
- `agent/env/`: Gymnasium-style environment, simulator, action/reward/state/wall/stone components.
- `agent/mcts/`: Monte Carlo tree search, nodes, [CEM](glossary.md#cem)/root proposal helpers.
- `agent/sampler/`: Ray-based MCTS episode sampler and h5 data generation.
- `planning/`: diffsim planner setup, grasp/IK/regrasp helpers, execution (manipulator-specific, to be rewritten), trajectory visualization.
- `perception/`: point-cloud/mesh/[DSF](glossary.md#dsf) reconstruction utilities; scan/scene-id runtime is manipulator/NUC-specific and to be redesigned.
- `ros2/`: ROS2 publisher/subscriber node wrappers and control helpers (manipulator-specific, to be rewritten for the manipulator).
- `gui/`: Open3D/PySide execution viewer (manipulator-specific execution UI, to be adapted).
- `loader/`: h5 and height-map dataset loaders.
- `model/`: stone/manipulator/gripper loading and URDF mesh transform helpers (manipulator/gripper models to be swapped for manipulator/end-effector models).
- `utils/`: geometry, neural-network, DSF, scheduling, data helpers.
- `scripts/desktop/`: manipulator planning/execution entrypoints; will need manipulator equivalents.
- `scripts/nuc/`: manipulator NUC-side scanning/pose-id scripts; **not applicable** to this project's desktop-only topology.
- `scripts/simulation/`, `scripts/train/`, `scripts/test/`, `scripts/debug/`: mostly hardware-agnostic simulation/training/debug tooling.
- `assets/`: stone, gripper, manipulator meshes/URDFs/PCDs/[DSF](glossary.md#dsf) assets; manipulator/gripper assets will need manipulator/end-effector equivalents.
- `docs/`: project documentation and LLM-facing wiki.

## Important Local Context

- Commands are normally run from the repository root.
- Imports assume the repository root is on `PYTHONPATH`.
- Use `direnv exec .` for every project command; see [Build and Run](build_and_run.md).
- There is no root package manifest (`pyproject.toml`, `setup.py`, `requirements.txt`, or `Makefile`).
- The sibling repo `../diffsim` builds the [`diffsimpy`](glossary.md#diffsimpy) Python extension used by this project (inherited dependency; expected to remain shared with `stacking-planner`).
- ROS2 usage for the manipulator (driver, message types, workspace layout) is **TBD**.
- This project is **desktop-only**: no onboard NUC, no Wi-Fi split between planning and perception machines.
- `docs/context.md` is the session log for this project; it was reset alongside this wiki on 2026-07-22.

## TODO

- Decide manipulator hardware (UR5e assumed but not confirmed) and end effector.
- Decide scene/pose identification approach (AprilTag vs. other) and validate feasibility for tabletop stones.
- Scope which manipulator-specific modules to delete vs. adapt vs. rewrite from scratch.
- Re-derive physical scale/tuning constants for tabletop stones and a manipulator workspace.
