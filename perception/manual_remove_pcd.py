import argparse
import pathlib
import shutil
import sys

import open3d as o3d

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from perception.reconstruction_dsf import manually_remove_points


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Open a single PCD file in the manual removal picker and write the "
            "filtered point cloud."
        )
    )
    parser.add_argument("pcd_path", help="Specific PCD file to edit.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PCD path. Defaults to <input-stem>_manual_removed.pcd unless "
            "--overwrite is set."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input PCD after writing a .bak copy next to it.",
    )
    parser.add_argument(
        "--distance",
        default=0.04,
        type=float,
        help="Radius removed around each manually picked point.",
    )
    parser.add_argument(
        "--max_remove_ratio",
        default=0.5,
        type=float,
        help="Skip deletion if selected regions exceed this fraction of the cloud.",
    )
    parser.add_argument(
        "--left_click",
        action="store_true",
        help="Pick removal centers with left click instead of pressing P.",
    )
    parser.add_argument(
        "--show_before",
        action="store_true",
        help="Show the input PCD in Open3D before manual removal.",
    )
    parser.add_argument(
        "--show_after",
        action="store_true",
        help="Show the filtered PCD in Open3D before saving.",
    )
    args = parser.parse_args()

    pcd_path = pathlib.Path(args.pcd_path)
    if not pcd_path.is_file():
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        raise ValueError(f"PCD file is empty or unreadable: {pcd_path}")

    if args.show_before:
        o3d.visualization.draw_geometries([pcd])

    filtered = manually_remove_points(
        pcd,
        distance=args.distance,
        max_remove_ratio=args.max_remove_ratio,
        left_clicking=args.left_click,
    )

    if args.show_after:
        o3d.visualization.draw_geometries([filtered])

    if args.overwrite:
        output_path = pcd_path
        backup_path = pcd_path.with_suffix(pcd_path.suffix + ".bak")
        shutil.copy2(pcd_path, backup_path)
        print(f"[backup] {pcd_path} -> {backup_path}")
    else:
        output_path = (
            pathlib.Path(args.output)
            if args.output is not None
            else pcd_path.with_name(f"{pcd_path.stem}_manual_removed{pcd_path.suffix}")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), filtered)
    print(
        f"[write] {output_path} "
        f"({len(pcd.points)} -> {len(filtered.points)} points)"
    )


if __name__ == "__main__":
    main()
