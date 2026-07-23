# Build and Run

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22. Keeps the environment facts that are still true for this
project; strips manipulator/NUC-specific commands until the manipulator system
is built. See `../stacking-planner/docs/llm/build_and_run.md` for the full
historical command reference from the manipulator project.

## Environment

- Development environment: direnv-based Python 3.12.3 environment in the `stacking` workspace (shared with `stacking-planner`).
- Use `direnv exec .` for every project command. Dependencies are provided by the direnv environment.
- Known project interpreter from the sibling project's docs:

```bash
/home/inrol-manipulator/worksapce/stacking/.direnv/python-3.12.3/bin/python
```

- Run commands from the repository root: `/home/inrol-manipulator/worksapce/stacking/stacking-tabletop`.
- The bare system `python3.12` may compile files but can miss dependencies such as `torch` or `h5py`.
- ROS2 usage (whether enabled, which driver/workspace for a UR5e) is **TBD** for this project; do not assume the manipulator's `~/ros2_ws` / `.envrc` ROS2 setup applies here without checking.

## Package and Dependency Status

- No root `pyproject.toml`, `setup.py`, `requirements.txt`, `environment.yml`, or `Makefile` observed (inherited from `stacking-planner`).
- Imports assume the repository root is on `PYTHONPATH`.
- [`diffsimpy`](glossary.md#diffsimpy) comes from the sibling repo `../diffsim`; expected to remain a shared dependency.
- `tp_msgs` (custom manipulator ROS2 message package) and manipulator-specific ROS2 packages are **not applicable** here; a manipulator project will need its own message/driver dependencies once decided.

## Main Configs (inherited, subject to change)

- `agent/configs/config.yml`
- `agent/configs/diffsim.yml`, `agent/configs/diffsim_fast.yml`
- `agent/configs/sampling.yml`
- `agent/configs/ablation*.yml`, `agent/configs/heightmap_*.yml`, `agent/configs/train_bc_mcts.yml`, `agent/configs/qfunction.yml`

These configs currently carry manipulator-scale target-wall geometry, gripper
constants, and field-tuned physics parameters inherited from
`stacking-planner`. Expect these to need retuning for tabletop-scale stones
and a manipulator workspace; do not assume current numeric defaults are
correct for this project.

## Commands

No commands are confirmed working/validated for this project yet post-reset.
`../stacking-planner/docs/llm/build_and_run.md` documents the full set of
manipulator-era commands (`generate_sequence`, `execute_offline`/`execute_online`,
`sceneid_from_logs`, ablation/training commands, etc.) — several of the
hardware-agnostic ones (training, sampling, ablation) are plausible starting
points to smoke-test the inherited planning core here, but have not yet been
re-verified against this repository's copy.

## TODO

- Verify which inherited commands (training/sampling/ablation, in particular) still run correctly in this repo as copied.
- Document the manipulator ROS2/driver setup once decided.
- Document first-time environment setup and official validation commands for this project.
