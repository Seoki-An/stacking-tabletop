# Experiments

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Use this file for concise experiment results that should help future agents.
Keep detailed logs in session directories or `docs/context.md`.

## Reset on 2026-07-22

This page was reset to empty alongside the rest of `docs/llm/` when
`stacking-tabletop` began its re-implementation as a manipulator-based
tabletop system. Historical MCTS ablation/training experiment results from
the manipulator project are in `../stacking-planner/docs/llm/experiments.md`
and `../stacking-planner/docs/context.md`; they concern the shared planning
core and may still be a useful reference for tuning direction, but were run
at manipulator/field scale and should not be assumed to transfer directly.

## Current Inferred Experiment Areas

None yet for this project. Populate as work starts, e.g.:

- Manipulator/end-effector grasp and motion planning tuning once a model exists.
- Tabletop-scale MCTS/CEM retuning.
- Scene/pose identification accuracy (AprilTag or other, once decided).

## TODO

- Add dates, commands, configs, data paths, and outcomes as experiments run in this project.
