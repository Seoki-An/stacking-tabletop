import argparse
import copy
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from perception.utils.refine_pcd import multiscale_icp


def _numeric_suffix(path: pathlib.Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else -1


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_theta = (np.trace(rotation) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


@dataclass
class PcdGroup:
    group_id: int
    pcds: list[o3d.geometry.PointCloud] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    merged: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)

    def add(self, pcd: o3d.geometry.PointCloud, frame_id: int, voxel: float):
        self.pcds.append(pcd)
        self.frame_ids.append(frame_id)
        self.merged += pcd
        if voxel > 0:
            self.merged = self.merged.voxel_down_sample(voxel)


def load_filtered_clouds(
    scan_dir: pathlib.Path,
    input_dir: str,
    include_rejected: bool = False,
    rejected_dir: str = "filtered_rejected",
    max_count: int = None,
):
    paths = list((scan_dir / input_dir).glob("*.pcd"))
    if include_rejected:
        paths.extend((scan_dir / rejected_dir).glob("*.pcd"))
    paths = sorted(paths, key=_numeric_suffix)
    if max_count is not None and max_count > 0:
        paths = paths[:max_count]

    entries = []
    for path in paths:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            continue
        entries.append((_numeric_suffix(path), path, pcd))
    return entries


def register_cloud_to_group(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_sizes,
    max_iters,
    min_fitness: float,
    min_correspondences: int,
    max_translation: float,
    max_rotation_deg: float,
):
    if len(source.points) < 30 or len(target.points) < 30:
        return np.eye(4), None, False

    transform, history = multiscale_icp(
        source,
        target,
        init_trans=np.eye(4),
        voxel_sizes=voxel_sizes,
        max_iters=max_iters,
        method="point_to_plane",
    )
    _, fitness, rmse, n_corr = history[-1]
    translation = np.linalg.norm(transform[:3, 3])
    rotation = _rotation_angle_deg(transform[:3, :3])
    metrics = {
        "fitness": fitness,
        "rmse": rmse,
        "n_corr": n_corr,
        "translation": translation,
        "rotation": rotation,
    }
    accepted = (
        fitness >= min_fitness
        and n_corr >= min_correspondences
        and translation <= max_translation
        and rotation <= max_rotation_deg
    )
    return transform, metrics, accepted


def build_incremental_groups(
    entries,
    group_voxel: float,
    min_fitness: float,
    min_correspondences: int,
    max_translation: float,
    max_rotation_deg: float,
    new_group_min_fitness: float = 0.05,
    new_group_min_correspondences: int = 30,
    verbose: bool = False,
):
    groups = []
    skipped = []
    voxel_sizes = [0.06, 0.03, 0.015]
    max_iters = [70, 40, 20]

    for frame_id, path, pcd in entries:
        if not groups:
            group = PcdGroup(group_id=0)
            group.add(copy.deepcopy(pcd), frame_id, group_voxel)
            groups.append(group)
            print(f"[grouping] frame={frame_id} -> new group 0")
            continue

        best = None
        for group in groups:
            transform, metrics, accepted = register_cloud_to_group(
                pcd,
                group.merged,
                voxel_sizes=voxel_sizes,
                max_iters=max_iters,
                min_fitness=min_fitness,
                min_correspondences=min_correspondences,
                max_translation=max_translation,
                max_rotation_deg=max_rotation_deg,
            )
            if metrics is None:
                continue
            score = (metrics["fitness"], metrics["n_corr"], -metrics["rmse"])
            if best is None or score > best["score"]:
                best = {
                    "group": group,
                    "transform": transform,
                    "metrics": metrics,
                    "accepted": accepted,
                    "score": score,
                }

        if best is not None and best["accepted"]:
            aligned = copy.deepcopy(pcd)
            aligned.transform(best["transform"])
            best["group"].add(aligned, frame_id, group_voxel)
            if verbose:
                m = best["metrics"]
                print(
                    f"[grouping] frame={frame_id} -> group {best['group'].group_id}: "
                    f"fitness={m['fitness']:.3f}, rmse={m['rmse']:.4f}, "
                    f"corr={m['n_corr']}, trans={m['translation']:.3f}, "
                    f"rot={m['rotation']:.1f}"
                )
        else:
            can_start_new_group = best is None or (
                best["metrics"]["fitness"] >= new_group_min_fitness
                and best["metrics"]["n_corr"] >= new_group_min_correspondences
            )
            if not can_start_new_group:
                skipped.append((frame_id, path, pcd, best["metrics"]))
                m = best["metrics"]
                print(
                    f"[grouping] frame={frame_id} -> skip "
                    f"(best group {best['group'].group_id}: "
                    f"fitness={m['fitness']:.3f}, corr={m['n_corr']}, "
                    f"trans={m['translation']:.3f}, rot={m['rotation']:.1f})"
                )
                continue

            group = PcdGroup(group_id=len(groups))
            group.add(copy.deepcopy(pcd), frame_id, group_voxel)
            groups.append(group)
            if best is None:
                print(f"[grouping] frame={frame_id} -> new group {group.group_id}")
            else:
                m = best["metrics"]
                print(
                    f"[grouping] frame={frame_id} -> new group {group.group_id} "
                    f"(best group {best['group'].group_id}: "
                    f"fitness={m['fitness']:.3f}, corr={m['n_corr']}, "
                    f"trans={m['translation']:.3f}, rot={m['rotation']:.1f})"
                )

    return groups, skipped


def merge_groups(
    groups,
    group_voxel: float,
    final_voxel: float,
    min_fitness: float,
    min_correspondences: int,
    max_translation: float,
    max_rotation_deg: float,
    verbose: bool = False,
):
    if not groups:
        return o3d.geometry.PointCloud(), []

    groups = sorted(groups, key=lambda group: len(group.frame_ids), reverse=True)
    final = copy.deepcopy(groups[0].merged)
    merged_group_ids = [groups[0].group_id]
    voxel_sizes = [0.10, 0.05, 0.025, 0.01]
    max_iters = [100, 60, 35, 20]

    for group in groups[1:]:
        transform, metrics, accepted = register_cloud_to_group(
            group.merged,
            final,
            voxel_sizes=voxel_sizes,
            max_iters=max_iters,
            min_fitness=min_fitness,
            min_correspondences=min_correspondences,
            max_translation=max_translation,
            max_rotation_deg=max_rotation_deg,
        )
        if metrics is None:
            continue
        print(
            f"[group merge] group {group.group_id} -> final: "
            f"fitness={metrics['fitness']:.3f}, rmse={metrics['rmse']:.4f}, "
            f"corr={metrics['n_corr']}, trans={metrics['translation']:.3f}, "
            f"rot={metrics['rotation']:.1f}, {'merge' if accepted else 'skip'}"
        )
        if not accepted:
            continue
        aligned = copy.deepcopy(group.merged)
        aligned.transform(transform)
        final += aligned
        if group_voxel > 0:
            final = final.voxel_down_sample(group_voxel)
        merged_group_ids.append(group.group_id)

    if final_voxel > 0 and len(final.points) > 0:
        final = final.voxel_down_sample(final_voxel)
    return final, merged_group_ids


def save_groups(scan_dir: pathlib.Path, groups, output_dir: str):
    group_dir = scan_dir / output_dir
    group_dir.mkdir(exist_ok=True)
    for group in groups:
        out_path = group_dir / f"group_{group.group_id:03d}.pcd"
        o3d.io.write_point_cloud(str(out_path), group.merged)
        frame_path = group_dir / f"group_{group.group_id:03d}_frames.txt"
        frame_path.write_text("\n".join(str(i) for i in group.frame_ids) + "\n")


def save_skipped(scan_dir: pathlib.Path, skipped, output_dir: str):
    if not skipped:
        return
    skipped_dir = scan_dir / output_dir
    skipped_dir.mkdir(exist_ok=True)
    lines = []
    for i, (frame_id, path, pcd, metrics) in enumerate(skipped):
        out_path = skipped_dir / f"skipped_{frame_id:04d}.pcd"
        o3d.io.write_point_cloud(str(out_path), pcd)
        lines.append(
            f"{frame_id},fitness={metrics['fitness']:.6f},"
            f"corr={metrics['n_corr']},trans={metrics['translation']:.6f},"
            f"rot={metrics['rotation']:.6f},source={path}"
        )
    (skipped_dir / "skipped_frames.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally group filtered PCDs by ICP consistency, then merge "
            "the resulting groups into one PCD."
        )
    )
    parser.add_argument("scan_dir", help="Directory containing filtered/*.pcd")
    parser.add_argument(
        "--input_dir",
        default="filtered",
        help="Input filtered cloud directory under scan_dir.",
    )
    parser.add_argument(
        "--include_rejected",
        action="store_true",
        help="Also include filtered_rejected/*.pcd in the incremental grouping.",
    )
    parser.add_argument(
        "--rejected_dir",
        default="filtered_rejected",
        help="Rejected filtered cloud directory under scan_dir.",
    )
    parser.add_argument(
        "--output",
        default="merged_groups.pcd",
        help="Output PCD filename under scan_dir.",
    )
    parser.add_argument(
        "--groups_output_dir",
        default="icp_groups",
        help="Directory under scan_dir where per-group PCDs are written.",
    )
    parser.add_argument(
        "--max_count",
        default=None,
        type=int,
        help="Load only the first N clouds for quick debugging.",
    )
    parser.add_argument(
        "--group_voxel",
        default=0.01,
        type=float,
        help="Voxel size used when updating group merged clouds.",
    )
    parser.add_argument(
        "--final_voxel",
        default=0.005,
        type=float,
        help="Voxel size for final merged cloud.",
    )
    parser.add_argument(
        "--frame_fitness_threshold",
        default=0.30,
        type=float,
        help="Minimum ICP fitness for adding one frame to an existing group.",
    )
    parser.add_argument(
        "--frame_min_correspondences",
        default=50,
        type=int,
        help="Minimum ICP correspondences for adding one frame to a group.",
    )
    parser.add_argument(
        "--frame_max_translation",
        default=0.30,
        type=float,
        help="Max frame-to-group ICP translation correction.",
    )
    parser.add_argument(
        "--frame_max_rotation_deg",
        default=25.0,
        type=float,
        help="Max frame-to-group ICP rotation correction.",
    )
    parser.add_argument(
        "--new_group_min_fitness",
        default=0.05,
        type=float,
        help="Minimum best-match ICP fitness required to start a new group.",
    )
    parser.add_argument(
        "--new_group_min_correspondences",
        default=30,
        type=int,
        help="Minimum best-match correspondences required to start a new group.",
    )
    parser.add_argument(
        "--group_fitness_threshold",
        default=0.25,
        type=float,
        help="Minimum ICP fitness for merging one group into final cloud.",
    )
    parser.add_argument(
        "--group_min_correspondences",
        default=100,
        type=int,
        help="Minimum ICP correspondences for merging one group into final cloud.",
    )
    parser.add_argument(
        "--group_max_translation",
        default=0.50,
        type=float,
        help="Max group-to-final ICP translation correction.",
    )
    parser.add_argument(
        "--group_max_rotation_deg",
        default=35.0,
        type=float,
        help="Max group-to-final ICP rotation correction.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    scan_dir = pathlib.Path(args.scan_dir)
    entries = load_filtered_clouds(
        scan_dir,
        input_dir=args.input_dir,
        include_rejected=args.include_rejected,
        rejected_dir=args.rejected_dir,
        max_count=args.max_count,
    )
    if not entries:
        raise RuntimeError(f"No PCDs found under {scan_dir / args.input_dir}")

    print(f"Loaded {len(entries)} filtered clouds.")
    groups, skipped = build_incremental_groups(
        entries,
        group_voxel=args.group_voxel,
        min_fitness=args.frame_fitness_threshold,
        min_correspondences=args.frame_min_correspondences,
        max_translation=args.frame_max_translation,
        max_rotation_deg=args.frame_max_rotation_deg,
        new_group_min_fitness=args.new_group_min_fitness,
        new_group_min_correspondences=args.new_group_min_correspondences,
        verbose=args.verbose,
    )
    print(
        "[grouping] result: "
        + ", ".join(
            f"group {g.group_id}: {len(g.frame_ids)} frames" for g in groups
        )
    )
    save_groups(scan_dir, groups, args.groups_output_dir)
    save_skipped(scan_dir, skipped, f"{args.groups_output_dir}_skipped")

    final, merged_group_ids = merge_groups(
        groups,
        group_voxel=args.group_voxel,
        final_voxel=args.final_voxel,
        min_fitness=args.group_fitness_threshold,
        min_correspondences=args.group_min_correspondences,
        max_translation=args.group_max_translation,
        max_rotation_deg=args.group_max_rotation_deg,
        verbose=args.verbose,
    )

    out_path = scan_dir / args.output
    o3d.io.write_point_cloud(str(out_path), final)
    print(
        f"Wrote {out_path} with {len(final.points)} points. "
        f"Merged groups: {merged_group_ids}"
    )

    if args.visualize:
        vis_groups = []
        colors = [
            [0.9, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.3, 1.0],
            [0.9, 0.7, 0.1],
            [0.7, 0.1, 0.9],
        ]
        for group in groups:
            pcd = copy.deepcopy(group.merged)
            pcd.paint_uniform_color(colors[group.group_id % len(colors)])
            vis_groups.append(pcd)
        o3d.visualization.draw_geometries(vis_groups)
        o3d.visualization.draw_geometries([final])


if __name__ == "__main__":
    main()
