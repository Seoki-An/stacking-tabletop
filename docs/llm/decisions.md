# Decisions

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Record stable design decisions here. Add new entries when changing behavior,
interfaces, workflow commands, or environment assumptions.

## Reset on 2026-07-22

This page was reset to empty as part of switching `stacking-tabletop` from an
manipulator-based to a manipulator-based (UR5e candidate) tabletop stacking
system, desktop-only (no NUC). The prior decisions log accumulated in
`stacking-planner` covers a large number of MCTS/diffsim tuning and
manipulator-execution decisions; many concern the shared planning core
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
- **Self-contained UR5e model URDF (2026-07-23):**
  `assets/ur5e/ur5e.urdf` is a flattened model derived from the sibling
  `Universal_Robots_ROS2_Description` checkout, so tabletop visualization and
  diffsim FK do not require ROS package lookup or xacro at runtime. Mesh paths
  are relative to the URDF. The upstream collision STL meshes are used as
  visual geometry because the current Open3D environment cannot load the
  upstream Collada files. Ordinary URDF `<collision><mesh>` elements are
  intentionally omitted: diffsim's planner requires its custom `dsf_vert`
  collision geometry, and treating an ordinary mesh as a collision geometry
  currently leaves a null collision body. Robot collision checking therefore
  remains pending generation of UR5e DSF assets.
- **SR gripper mounting on UR5e (2026-07-23):** The full visualization model
  mounts `sr_gripper_base_link` rigidly and coincident with UR5e `tool0`
  (`xyz="0 0 0"`, `rpy="0 0 0"`). The gripper extends along the tool frame's
  local +Z axis. `assets/ur5e/ur5e.urdf` contains all four finger joints,
  mirroring the excavator's full arm-plus-gripper visualization URDF; the
  viewer synchronizes those joints through one gripper control using
  `[left_1, left_2, right_1, right_2] = [theta, -theta, theta, -theta]`.
  This makes the full model 10-DOF, so it is not the six-axis diffsim planning model.
  Preparing the end-effector-only `ur5e_ik.urdf` and SR-gripper `dsf_vert`
  collision model remains separate work.
- **Principal-axis re-alignment of placement orientation (2026-07-23):** In the
  default grid sampling path (`PlanarSampler._mixed_grid_poses` else-branch),
  the base placement orientation is no longer the raw asset frame. Each stone is
  first re-aligned so its longest principal axis → world X, middle → Y, shortest
  → Z, then the configured `action.rotation.angles_x`/`angles_z` are applied on
  top (world-frame, `R_angle * R_align`). Principal axes come from PCA of the
  stone's convex-hull vertices, computed and cached by
  `StoneObject.principal_axis_alignment()`. Scope is intentionally limited to the
  general grid path; the floor-fill/face-normal (wall-inward) and scan-pose
  orientation paths are unchanged. Fallback is identity when no stone index is
  available (the non-score path).

## TODO

- As re-implementation proceeds, port over/re-confirm the still-applicable core planning decisions from `stacking-planner` (data contracts, simulation terminology, MCTS tree-preservation semantics) with a note that they were re-confirmed rather than assumed.
