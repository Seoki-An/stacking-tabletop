import argparse
import pathlib
import re
import shutil


def _stone_id_from_dir(path: pathlib.Path) -> str:
    matches = re.findall(r"\d+", path.name)
    return matches[-1] if matches else path.name


def export_reconstructed_assets(
    root_dir: pathlib.Path,
    asset_dir: pathlib.Path,
    name_style: str = "model_id",
    overwrite: bool = False,
    prefer_multiple_dsf: bool = False,
):
    dsf_names = (
        ("dsf_multiple.obj", "dsf.obj")
        if prefer_multiple_dsf
        else ("dsf.obj", "dsf_multiple.obj")
    )
    scan_dirs = sorted(
        {
            path.parent
            for dsf_name in dsf_names
            for path in root_dir.rglob(dsf_name)
            if (path.parent / "merged_refined.pcd").is_file()
        },
        key=lambda p: (_stone_id_from_dir(p), str(p)),
    )

    if not scan_dirs:
        dsf_desc = " or ".join(dsf_names)
        print(
            f"No directories with both {dsf_desc} and merged_refined.pcd "
            f"under {root_dir}"
        )
        return

    asset_dir.mkdir(parents=True, exist_ok=True)
    for scan_dir in scan_dirs:
        stone_id = _stone_id_from_dir(scan_dir)
        if name_style in ("model_id", "stone_merged"):
            stem = f"model_{stone_id}"
        else:
            stem = scan_dir.name
        obj_src = next(
            (scan_dir / name for name in dsf_names if (scan_dir / name).is_file()), None
        )
        if obj_src is None:
            continue
        pcd_src = scan_dir / "merged_refined.pcd"
        mesh_src = scan_dir / "mesh_refined.ply"
        obj_dst = asset_dir / f"{stem}.obj"
        pcd_dst = asset_dir / f"{stem}.pcd"
        mesh_dst = asset_dir / f"{stem}_mesh.ply"

        for src, dst in [(obj_src, obj_dst), (pcd_src, pcd_dst), (mesh_src, mesh_dst)]:
            if dst.exists() and not overwrite:
                print(f"[skip] {dst} already exists. Use --overwrite to replace it.")
                continue
            shutil.copy2(src, dst)
            print(f"[copy] {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy reconstructed dsf.obj and merged_refined.pcd files into an "
            "assets/stone-style directory as model_<parent-id>.obj/.pcd."
        )
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing reconstructed stone folders.",
    )
    parser.add_argument(
        "--asset_dir",
        default="assets/stone",
        help="Destination asset directory.",
    )
    parser.add_argument(
        "--name_style",
        choices=["model_id", "parent", "stone_merged"],
        default="model_id",
        help=(
            "'model_id' writes model_<parent-number>; "
            "'parent' writes <parent-dir-name>; "
            "'stone_merged' reads stone<num>_merged folders and writes model_<num>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model_<id>.obj/.pcd files.",
    )
    parser.add_argument(
        "--prefer_multiple_dsf",
        action="store_true",
        help="Use dsf_multiple.obj instead of dsf.obj when both are present.",
    )
    args = parser.parse_args()

    export_reconstructed_assets(
        pathlib.Path(args.root_dir),
        pathlib.Path(args.asset_dir),
        name_style=args.name_style,
        overwrite=args.overwrite,
        prefer_multiple_dsf=args.prefer_multiple_dsf,
    )


if __name__ == "__main__":
    main()
