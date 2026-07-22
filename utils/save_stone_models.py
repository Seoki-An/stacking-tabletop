import os
import re
import shutil
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Save stone models")
    argparser.add_argument(
        "root_dir",
        help="Root directory to start search (ex: STONE_PCD)",
        default="data/stone_pcd",
    )
    argparser.add_argument(
        "save_dir",
        help="Directory to save the stone models (ex: assets/stone)",
        default="assets/stone",
    )
    args = argparser.parse_args()
    root_dir = os.path.abspath(args.root_dir)
    save_dir = os.path.abspath(args.save_dir)
    if not os.path.isdir(root_dir):
        print(f"Error: '{args.root_dir}' is not available.")
        exit(1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    subdirs = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]
    for name in subdirs:
        num = int(re.findall(r"\d+", name)[0])
        print(f"Processing: stone {num}")
        pcd_file = os.path.join(root_dir, name, "merged_refined.pcd")
        pcd_save_file = os.path.join(save_dir, f"model_{num}.pcd")
        shutil.copy(pcd_file, pcd_save_file)

        obj_file = os.path.join(root_dir, name, "dsf.obj")
        obj_save_file = os.path.join(save_dir, f"model_{num}.obj")
        shutil.copy(obj_file, obj_save_file)
