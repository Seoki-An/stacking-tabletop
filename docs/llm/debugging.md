# Debugging

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22. Excavator/NUC-specific gotchas were removed; see
`../stacking-planner/docs/llm/debugging.md` for that historical content. This
page starts with only the gotchas that are not tied to manipulator hardware.

## General Gotchas

- Run scripts from the repository root; many paths are relative.
- Use `direnv exec .` for every project command. The system Python may miss project dependencies.
- Open3D GUI/offscreen rendering may need a working EGL/OSMesa/GL setup.
- Check `CUDA_VISIBLE_DEVICES` before long training/planning jobs.

## `diffsimpy` and Sibling Repo Issues

If Python fails with a missing method or field on [`diffsimpy`](glossary.md#diffsimpy), check the corresponding pybind file in `../diffsim/interop/python/src/`. The C++ class may exist without being exposed to Python.

If a config field default or type looks wrong, check C++ config headers under `../diffsim/src/*/config.hpp`.

Important Python-facing submodules:

- `diffsimpy.diffsim`
- [`diffsimpy.posegen`](glossary.md#posegen)
- [`diffsimpy.poseinit`](glossary.md#poseinit)
- `diffsimpy.planner`
- `diffsimpy.sceneid` (used by the manipulator SceneID flow; relevance to this project's perception redesign is TBD)

## TODO

- Re-populate this page with manipulator/perception-specific gotchas as the new system is built.
- Confirm whether any of the manipulator-era diffsim/motion-planning gotchas (grasp initialization, contact tuning) still apply once a manipulator end-effector model exists.
