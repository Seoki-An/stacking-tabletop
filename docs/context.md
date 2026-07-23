# Codex Project Context

Reset on 2026-07-22. This file previously carried the full manipulator-project
session log inherited from `stacking-planner` (~4,100 lines of MCTS/execution
history). That log described a different robot, a different compute
topology, and a perception stack this project is not using; keeping it here
would mislead future sessions into treating stale manipulator specifics as
current fact. It has been cleared. The original remains available at
`../stacking-planner/docs/context.md` for historical reference.

## What This Repository Is

`stacking-tabletop` starts as a full copy of `../stacking-planner` (a Python
robotics/ML stack for planning and executing stone stacking with an
manipulator and bucket-style gripper). It is being re-implemented into a
**manipulator-based tabletop stacking system**:

- Robot: manipulator arm (UR5e is the current candidate, not yet confirmed), not an manipulator.
- Compute topology: **desktop only** — no onboard NUC, no Wi-Fi split between planning and perception machines. The desktop talks directly to the manipulator and the cameras.
- Scene/pose identification: to be redesigned. AprilTag fiducials are the leading candidate per the user's initial framing, but this is explicitly open ("should be considered later") — no perception approach has been implemented yet.

See `docs/llm/overview.md` for the full breakdown of what is inherited as-is
vs. what needs a fresh design, and the rest of `docs/llm/` for the reset
wiki pages per subsystem.

There is no package manifest in the repo root, matching `stacking-planner`.
Imports assume the repo root is on `PYTHONPATH` and commands are normally run
from `/home/inrol-manipulator/worksapce/stacking/stacking-tabletop`.

## 2026-07-22 Kickoff

- Repository copied from `stacking-planner` as the starting point for this re-implementation.
- `docs/llm/` wiki and this file were reset to remove manipulator/NUC-specific content that no longer applies; hardware-agnostic planning-core vocabulary and config semantics were kept.
- No manipulator hardware, end-effector, perception approach, or ROS2 control interface has been decided or implemented yet.

## TODO

- Confirm manipulator (UR5e assumption) and end effector.
- Decide and implement the scene/pose identification approach.
- Scope which manipulator-specific modules (`ros2/`, `planning/execution*.py`, `gui/execution_window.py`, `scripts/nuc/`, `scripts/desktop/execute_*.py`) get deleted vs. adapted vs. rewritten.
- Re-derive physical scale/tuning constants for tabletop stones and a manipulator workspace.
