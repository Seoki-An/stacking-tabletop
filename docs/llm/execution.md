# Execution

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22. This page is intentionally a stub: the field execution
workflow is being redesigned for a manipulator-based, desktop-only system and
has not been implemented yet.

## Status

The prior `stacking-planner` execution stack (`scripts/desktop/execute_offline.py`,
`scripts/desktop/execute_online.py`, `planning/execution*.py`,
`gui/execution_window.py`) was built around:

- an manipulator with a bucket-style gripper, controlled via ROS2 joint/phase topics,
- a multi-process split (Open3D/PySide GUI in the main process, planning/ROS2/Ray/diffsim in a worker process) to keep long native calls off the GUI thread,
- a NUC that performed in-hand/field/scene LiDAR scans on request and streamed results back to the desktop over Wi-Fi,
- an operator-in-the-loop review GUI at each pick/place/scan step.

None of the ROS2 manipulator-phase choreography or NUC handoff applies to this
project. It is documented in `../stacking-planner/docs/llm/execution.md` for
reference only; the manipulator control loop, operator review UX, and
process topology all need to be redesigned from scratch.

## Open Questions For This Project

- Manipulator control interface: ROS2 driver (e.g. `ur_robot_driver`) vs. another control path — undecided.
- Whether the GUI/worker multi-process split is still warranted for a (presumably lighter) manipulator control loop, or whether execution can run in a single process.
- What operator-in-the-loop review steps are still needed (motion preview/accept, place confirmation, etc.) given the new hardware and perception approach.
- Whether "online" (plan-one-step-at-a-time) execution and "offline" (precomputed full sequence) execution are both still needed, as in the manipulator system.

## Related Files (inherited, pending redesign)

- `scripts/desktop/execute_offline.py`, `scripts/desktop/execute_online.py`: manipulator execution entrypoints; not applicable as-is.
- `planning/execution.py` and `planning/execution_*.py` mixins: manipulator execution worker logic; likely needs a ground-up rewrite for manipulator control.
- `gui/execution_window.py`, `gui/viewer.py`, `gui/window.py`: Open3D/PySide execution viewer; UI structure may be reusable, control plumbing is not.
- `ros2/*`: manipulator joint/phase/pose/pcd node wrappers; not applicable to a manipulator control interface.
- `docs/llm/planning.md`
- `docs/llm/perception.md`

## TODO

- Decide the manipulator control interface.
- Design the desktop-only execution loop (no NUC handoff).
- Once decided, write the actual execution architecture here (replacing this stub).
