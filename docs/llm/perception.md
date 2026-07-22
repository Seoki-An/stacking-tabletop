# Perception

Wiki: [Overview](overview.md) | [Architecture](architecture.md) | [Planning](planning.md) | [Execution](execution.md) | [Perception](perception.md) | [Build and Run](build_and_run.md) | [Debugging](debugging.md) | [Decisions](decisions.md) | [Experiments](experiments.md) | [Glossary](glossary.md)

Reset on 2026-07-22. This page is intentionally a stub: scene identification
and pose identification are being redesigned for this project and have not
been implemented yet.

## Status

The prior `stacking-planner` perception stack was built around:

- an onboard NUC running LiDAR scans and streaming point clouds to the desktop over Wi-Fi,
- Open3D ICP + a custom SceneID solver (`diffsimpy.sceneid`) to identify placed-stone poses from scene point clouds,
- a separate offline reconstruction pipeline (`perception/reconstruction_*`, `perception/dsf_fit/`) that builds stone mesh/DSF models from in-hand scans.

None of the NUC/ICP/SceneID scanning workflow applies to this project's
desktop-only topology. It is documented in
`../stacking-planner/docs/llm/perception.md` for reference only.

## Open Questions For This Project

- **Sensing**: which camera(s) are used, and where are they mounted (fixed overhead, wrist-mounted, or both)?
- **Pose identification method**: AprilTag fiducials are the leading candidate (per the user's initial framing) but this is explicitly undecided — "should be considered later." Alternatives (marker-free pose estimation, point-cloud registration reused from the existing pipeline) have not been ruled out.
- **Stone model reconstruction**: `perception/reconstruction_mesh.py`, `perception/reconstruction_dsf_multiple.py`, and `perception/dsf_fit/` build mesh/DSF stone models from point clouds and may still be reusable if the new camera setup still produces per-stone point clouds; unconfirmed.
- **Scene scan cadence**: the old system re-scanned the scene after every placement to correct for physical drift. Whether that pattern is still needed (and how expensive it is with the new sensing approach) is undecided.

## Related Files (inherited, pending redesign)

- `perception/get_stone_pcd.py`, `perception/merge_stone_pcd_groups.py`, `perception/merge_filtered_groups.py`: in-hand PCD merging utilities tied to the old NUC scan format.
- `perception/reconstruction_mesh.py`, `perception/reconstruction_dsf_multiple.py`, `perception/dsf_fit/`: stone model reconstruction; candidate for reuse.
- `perception/sceneid_runtime.py`: SceneID runtime entrypoint used by the old excavator execution flow; not applicable as-is.
- `scripts/nuc/*`: NUC-side scanning/pose-identification scripts; **not applicable** to a desktop-only topology. Any needed logic will be ported into a desktop-side perception module rather than kept as `scripts/nuc/`.

## TODO

- Decide the sensing/pose-identification approach.
- Once decided, write the actual perception architecture here (replacing this stub).
- Determine whether `perception/reconstruction_*` and `perception/dsf_fit/` are reused, adapted, or replaced.
