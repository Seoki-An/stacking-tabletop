import os
import argparse
import re

import open3d as o3d
import numpy as np
import pandas as pd

from model import get_excavator_model, update_urdf_mesh
from perception import box_crop_largest_cluster


def available_scan_indices(scan_dir, num_joint_rows):
    pattern = re.compile(r"scene_scan_lidar[123]_(\d+)_raw\.pcd$")
    indices = set()
    for name in os.listdir(scan_dir):
        match = pattern.match(name)
        if match:
            indices.add(int(match.group(1)))
    return sorted(i for i in indices if i < num_joint_rows)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", type=str, default=".data/scene_pcd/260402_9")
    argparser.add_argument(
        "--rot-offset-deg",
        type=float,
        default=0.0,
        help=(
            "Rotation offset in degrees added to joint 0 before transforming "
            "raw LiDAR PCDs."
        ),
    )
    args = argparser.parse_args()
    rot_offset = np.deg2rad(args.rot_offset_deg)

    model, meshes = get_excavator_model()

    dir = args.dir
    csv_path = os.path.join(dir, "joint_log.csv")
    csv_file = pd.read_csv(csv_path, header=None, skipinitialspace=False)
    scan_indices = available_scan_indices(dir, len(csv_file))
    if not scan_indices:
        raise FileNotFoundError(f"No raw scene_scan_lidar*_N_raw.pcd files in {dir}")
    print(f"Loading {len(scan_indices)} scan poses: {scan_indices}")

    scene_pcd = o3d.geometry.PointCloud()
    lidar_list = []
    mesh_list = []
    for i in scan_indices:
        q = np.asarray(csv_file.iloc[i]).copy()

        q = np.concatenate([q, np.zeros(2)])
        q[0] += rot_offset
        model.SetState(q)

        for lidar_idx in (1, 2, 3):
            pcd_path = os.path.join(dir, f"scene_scan_lidar{lidar_idx}_{i}_raw.pcd")
            if not os.path.exists(pcd_path):
                print(f"Skipping missing raw PCD: {pcd_path}")
                continue

            pcd = o3d.io.read_point_cloud(pcd_path)
            if pcd.is_empty():
                print(f"Skipping empty raw PCD: {pcd_path}")
                continue

            link_name = f"lidar{lidar_idx}_link"
            lidar_frame = np.eye(4)
            lidar_frame[:3, :3] = model.GetLink(link_name).GetRotation()
            lidar_frame[:3, -1] = model.GetLink(link_name).GetPosition()
            pcd.transform(lidar_frame)
            scene_pcd += pcd

            lidar = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0]
            )
            lidar.transform(lidar_frame)
            lidar_list.append(lidar)

        meshes_t = update_urdf_mesh(model, meshes, q)
        mesh_list.extend(meshes_t.values())
        # o3d.visualization.draw_geometries([pcd2] + [mesh for mesh in meshes_t.values()])\

    # scene_pcd = box_crop_largest_cluster(
    #     scene_pcd, [0, 8], [-5, 5], [-1, 2], cluster=False
    # )

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10.0, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [scene_pcd] + mesh_list + [coord_frame] + lidar_list
    )

    o3d.io.write_point_cloud(os.path.join(dir, "merged_lidar.pcd"), scene_pcd)
