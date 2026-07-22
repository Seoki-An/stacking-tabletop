# Glossary

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22: excavator/NUC-specific terms removed or annotated;
hardware-agnostic planning vocabulary kept since it is expected to carry over
from `stacking-planner`. See `../stacking-planner/docs/llm/glossary.md` for
the full historical glossary.

## Action

A candidate stone placement, usually including stone id, initial pose, final
pose, support/posegen scores, and validation metadata.

## AprilTag

Fiducial marker system being considered (not yet decided or implemented) as
the basis for scene/pose identification in this project, replacing the
excavator project's LiDAR + ICP/SceneID approach. See [Perception](perception.md).

## BC-MCTS

Behavior cloning from [MCTS](#mcts)-generated data, used for offline training.

## CEM

Cross-Entropy Method; used to refine root [action](#action) proposals in [MCTS](#mcts).

## DSF

Differentiable support-function-style geometry representation used by
diffsim/[posegen](#posegen) assets and fitting workflows.

## diffsimpy

Python extension built from the sibling `../diffsim` repo.

## Field Scan / Scene Scan

In the excavator project, a scan of the pick area (field scan) or the placed
structure (scene scan). Whether this vocabulary carries over depends on the
new perception approach; see [Perception](perception.md).

## Grasp

Gripper pose/opening-angle candidate for picking or holding a stone. The
excavator gripper model is not applicable to a manipulator end effector;
grasp generation will need a new/updated end-effector model.

## Height Map

Orthographic scene/target depth representation used in observations and
height-map models.

## MCTS

Monte Carlo Tree Search used for stacking action selection.

## OmegaConf

Config system used by `agent/configs/*.yml` and training/planning scripts.

## Pick Plane / Place Plane

Support plane used for pick-side / place-side planner context.

## posegen

[`diffsimpy`](#diffsimpy) module used for pose optimization/contact solving.

## poseinit

[`diffsimpy`](#diffsimpy) module for pose initialization.

## Regrasp

Motion strategy that may put down and re-grasp a stone before final placement.

## ROS2

Robot Operating System 2. Used by the excavator project for joint/pose/phase/
point-cloud/status communication with an onboard NUC. Whether and how this
project uses ROS2 for the manipulator (e.g. a UR5e driver) is **TBD**; do not
assume the excavator project's node/topic layout applies.

## SceneID

`diffsimpy`/script workflow used by the excavator project to identify placed
stone poses from LiDAR scene point clouds. Not confirmed applicable to this
project; see [Perception](perception.md).

## State

Environment state, including stone set/sequence/poses and history.

## Target Wall

Desired target structure geometry represented by wall dimensions, origin,
taper, and meshes.

## UR5e

Candidate manipulator for this project (not yet confirmed). See [Overview](overview.md).

## TODO

- Confirm the manipulator/end-effector and add its terminology once decided.
- Add perception-stack vocabulary (AprilTag or otherwise) once the approach is decided.
- Add ROS2/topic glossary once the manipulator control interface is decided.
