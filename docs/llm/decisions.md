# Decisions

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Record stable design decisions here. Add new entries when changing behavior,
interfaces, workflow commands, or environment assumptions.

## Reset on 2026-07-22

This page was reset to empty as part of switching `stacking-tabletop` from an
excavator-based to a manipulator-based (UR5e candidate) tabletop stacking
system, desktop-only (no NUC). The prior decisions log accumulated in
`stacking-planner` covers a large number of MCTS/diffsim tuning and
excavator-execution decisions; many concern the shared planning core
([MCTS](glossary.md#mcts), [posegen](glossary.md#posegen), simulation
terminology, data contracts) and remain valid reference material, while
others (gripper tuning, NUC/ROS2 execution behavior, field-scale physical
constants) do not apply to this project.

See `../stacking-planner/docs/llm/decisions.md` for that full historical
record. Re-adopt individual decisions here explicitly, with a dated entry,
once they are confirmed to still hold for this project rather than assuming
they carry over silently.

## Decided So Far For This Project

- Desktop-only compute topology: no onboard NUC leg; the desktop talks directly to the manipulator and cameras.

## TODO

- As re-implementation proceeds, port over/re-confirm the still-applicable core planning decisions from `stacking-planner` (data contracts, simulation terminology, MCTS tree-preservation semantics) with a note that they were re-confirmed rather than assumed.
