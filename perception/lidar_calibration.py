import os
import sys
import argparse
import pathlib
import h5py
import numpy as np

from utils.se3_handle import invert_homogeneous, optimize_se3_from_batch


def main():
    parser = argparse.ArgumentParser(
        description="Merge .pcd files in the assigned root directory and its subdirectories."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory to start search (ex: STONE_PCD)",
        default=".data",
    )

    args = parser.parse_args()
    root_dir = pathlib.Path(args.root_dir)
    if not root_dir.is_dir():
        print(f"Error: '{args.root_dir}' is not available.")
        return

    dataset = {}
    h5_paths = list(root_dir.rglob("*.h5"))
    for h5_path in h5_paths:
        with h5py.File(str(h5_path), "r") as file:
            for key, val in file.items():
                if key in dataset.keys():
                    if val[()].size > 0:
                        dataset[key].append(val[()])
                else:
                    dataset[key] = [val[()]]
    for key, val in dataset.items():
        dataset[key] = np.concatenate(val, axis=0)

    T_bg = dataset["gripper_pose"]
    T_lg = dataset["gripper_pose_from_lidar"]
    T_bp = dataset["lidar_parent_pose"]
    T_pl_init = np.linalg.inv(T_bp[0]) @ dataset["lidar_pose"][0]
    T_bp_inv = invert_homogeneous(T_bp)
    T_lg_inv = invert_homogeneous(T_lg)

    T_pl = T_bp_inv @ T_bg @ T_lg_inv
    X_opt, info = optimize_se3_from_batch(
        T_pl_np=T_pl,
        T_pl_init=T_pl_init,
        device="cuda:0",
        lr=0.00002,
        epochs=60000,
    )
    print(X_opt)
    print(f"rpy: {info['rpy']}")
    print(f"trans: {info['trans']}")


if __name__ == "__main__":

    if sys.gettrace():
        sys.argv = [
            __file__,
            ".data",
        ]

    main()
