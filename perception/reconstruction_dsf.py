import os
import sys
import argparse
import open3d as o3d
import pathlib

from omegaconf import OmegaConf
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import perception.dsf_fit as dsf_fit
from perception.utils.reconstruction import (
    compute_hull_inertial_parameters,
    remove_detached_clusters,
    remove_radial_fringe_points,
)
from utils.wavefront import WavefrontExporter


def manually_remove_points(
    pcd: o3d.geometry.PointCloud,
    distance: float = 0.04,
    max_remove_ratio: float = 0.5,
    left_clicking: bool = False,
) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    picked_points = []

    plotter = pv.Plotter()
    cloud = pv.PolyData(points)
    cloud["z"] = points[:, 2]
    plotter.add_points(
        cloud,
        scalars="z",
        point_size=4,
        render_points_as_spheres=False,
        name="pcd",
    )
    plotter.add_text(
        "Manual removal\n"
        "Press P over unwanted islands/fringes to mark removal centers.\n"
        "Close the window to apply deletion.",
        position="upper_left",
        font_size=10,
    )

    def on_pick(point):
        if point is None:
            return
        point = np.asarray(point)
        picked_points.append(point)
        plotter.add_mesh(
            pv.Sphere(radius=distance, center=point),
            color="red",
            opacity=0.25,
            name=f"remove_region_{len(picked_points)}",
            pickable=False,
        )
        print(
            f"[manual] picked remove center #{len(picked_points)}: " f"{point.tolist()}"
        )

    plotter.enable_point_picking(
        callback=on_pick,
        show_message=(
            "Press P with the mouse over unwanted points. "
            "Close the window when done."
        ),
        left_clicking=left_clicking,
        show_point=True,
        point_size=12,
        color="red",
    )
    plotter.show()

    if not picked_points:
        print("[manual] No picked removal centers; keeping cloud unchanged.")
        return pcd

    picked_points = np.asarray(picked_points)
    tree = cKDTree(picked_points)
    dists, _ = tree.query(points, k=1)
    remove_mask = dists <= distance

    n_remove = int(remove_mask.sum())
    if n_remove == 0:
        print(
            "[manual] No points inside picked removal radius; keeping cloud unchanged."
        )
        return pcd

    remove_ratio = n_remove / len(remove_mask)
    if remove_ratio > max_remove_ratio:
        print(
            f"[manual] Selection covers {remove_ratio:.1%} of points; "
            "skipping to avoid deleting the stone by accident."
        )
        return pcd

    print(f"[manual] Removing {n_remove} / {len(remove_mask)} points.")
    return pcd.select_by_index(np.where(remove_mask)[0], invert=True)


def choose_reconstruct_pcd(
    pcd_file: pathlib.Path,
    pcd: o3d.geometry.PointCloud,
) -> bool:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return False

    selected = {"reconstruct": True}
    plotter = pv.Plotter(title=f"Reconstruct PCD: {pcd_file.name}")
    cloud = pv.PolyData(points)
    cloud["z"] = points[:, 2]
    plotter.add_points(
        cloud,
        scalars="z",
        point_size=4,
        render_points_as_spheres=False,
        name="pcd",
    )
    plotter.add_text(
        f"{pcd_file.name}\n\n"
        "Enter / R: reconstruct\n"
        "S / Delete: skip this PCD",
        position="upper_left",
        font_size=10,
    )

    def on_key_press(obj, _event):
        key = obj.GetKeySym()
        if key in ("Return", "space", "r", "R"):
            selected["reconstruct"] = True
            obj.TerminateApp()
        elif key in ("s", "S", "Delete"):
            selected["reconstruct"] = False
            obj.TerminateApp()

    plotter.iren.interactor.AddObserver("KeyPressEvent", on_key_press)
    plotter.show()
    plotter.close()

    return selected["reconstruct"]


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "root_dir",
        help="Root directory to start search (ex: STONE_PCD)",
        default="data/stone_pcd",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Whether to visualize the results."
    )
    parser.add_argument(
        "--show_result",
        action="store_true",
        help="Whether to show the final result.",
    )
    parser.add_argument(
        "--disable_statistical_outlier",
        action="store_true",
        help="Skip Open3D statistical outlier removal.",
    )
    parser.add_argument(
        "--stat_nb_neighbors",
        default=20,
        type=int,
        help="Neighbor count for statistical outlier removal.",
    )
    parser.add_argument(
        "--stat_std_ratio",
        default=2.0,
        type=float,
        help="Higher values preserve more sparse surface points.",
    )
    parser.add_argument(
        "--disable_detached_cluster_filter",
        action="store_true",
        help="Skip removal of small disconnected point islands.",
    )
    parser.add_argument(
        "--detached_verbose",
        action="store_true",
        help="Print detached-cluster keep/remove diagnostics.",
    )
    parser.add_argument(
        "--detached_voxel",
        default=0.01,
        type=float,
        help="Voxel size used only for detached-cluster labeling.",
    )
    parser.add_argument(
        "--detached_eps",
        default=0.035,
        type=float,
        help="DBSCAN eps for detached-cluster labeling.",
    )
    parser.add_argument(
        "--detached_min_points",
        default=6,
        type=int,
        help="DBSCAN min_points for detached-cluster labeling.",
    )
    parser.add_argument(
        "--detached_attach_distance",
        default=0.08,
        type=float,
        help="Keep split components this close to the largest component.",
    )
    parser.add_argument(
        "--detached_min_attached_size",
        default=80,
        type=int,
        help="Minimum downsampled size for a nearby split component to be preserved.",
    )
    parser.add_argument(
        "--detached_keep_ratio",
        default=0.10,
        type=float,
        help="Keep components this large relative to the largest component.",
    )
    parser.add_argument(
        "--disable_radial_fringe_filter",
        action="store_true",
        help="Skip radial outward fringe removal.",
    )
    parser.add_argument(
        "--fringe_k_neighbors",
        default=80,
        type=int,
        help="Directional neighbors used to estimate local radial surface envelope.",
    )
    parser.add_argument(
        "--fringe_margin",
        default=0.04,
        type=float,
        help="Allowed outward distance beyond the local radial envelope.",
    )
    parser.add_argument(
        "--fringe_reference_quantile",
        default=0.65,
        type=float,
        help="Local radial quantile used as the surface envelope.",
    )
    parser.add_argument(
        "--disable_cluster_filter",
        action="store_true",
        help="Skip DBSCAN clustering after voxel downsampling to preserve only the largest cluster.",
    )
    parser.add_argument(
        "--cluster_eps",
        default=0.05,
        type=float,
        help="DBSCAN eps after voxel downsampling. Larger values preserve coarse surfaces.",
    )
    parser.add_argument(
        "--cluster_min_points",
        default=8,
        type=int,
        help="DBSCAN min_points after voxel downsampling.",
    )
    parser.add_argument(
        "--manual_remove",
        action="store_true",
        help="Open an interactive PyVista picker before DSF fitting to remove points.",
    )
    parser.add_argument(
        "--manual_remove_distance",
        default=0.04,
        type=float,
        help="Radius removed around each manually picked point.",
    )
    parser.add_argument(
        "--manual_max_remove_ratio",
        default=0.5,
        type=float,
        help="Skip manual deletion if the selection exceeds this fraction of the cloud.",
    )
    parser.add_argument(
        "--manual_left_click",
        action="store_true",
        help="Pick removal centers with left click instead of pressing P.",
    )
    parser.add_argument(
        "--get_refined",
        action="store_true",
        help="Whether to save the refined point cloud after filtering.",
    )
    parser.add_argument(
        "--not_save_refined",
        action="store_true",
        help="Whether to skip saving the refined point cloud after filtering.",
    )
    parser.add_argument("--mu", default=1.0, type=float)
    parser.add_argument(
        "--dsf_num_vertices",
        default=None,
        type=int,
        help="Number of vertices in the fitted DSF. Defaults to dsf_fit/config.yml.",
    )

    args = parser.parse_args()
    visualize = args.visualize
    root_dir = pathlib.Path(args.root_dir)
    if not root_dir.is_dir():
        print(f"Error: '{args.root_dir}' is not available.")
        return
    if args.get_refined:
        pcd_paths = list(root_dir.rglob("merged_refined.pcd"))
    else:
        pcd_paths = list(root_dir.rglob("merged.pcd"))
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dsf_cfg = OmegaConf.load(os.path.join(cur_dir, "dsf_fit/config.yml"))
    if args.dsf_num_vertices is not None:
        dsf_cfg.optimizer.num_vertices = args.dsf_num_vertices

    for pcd_file in pcd_paths:
        print(f"Processing: {pcd_file}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        if visualize:
            if not choose_reconstruct_pcd(pcd_file, pcd):
                print(f"Skipping {pcd_file}: user selected skip.")
                continue
        if not args.disable_statistical_outlier:
            pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=args.stat_nb_neighbors,
                std_ratio=args.stat_std_ratio,
            )
        if not args.disable_detached_cluster_filter:
            pcd = remove_detached_clusters(
                pcd,
                voxel=args.detached_voxel,
                eps=args.detached_eps,
                min_points=args.detached_min_points,
                attach_distance=args.detached_attach_distance,
                min_attached_size=args.detached_min_attached_size,
                keep_ratio=args.detached_keep_ratio,
                verbose=args.detached_verbose,
            )
        if not args.disable_radial_fringe_filter:
            pcd = remove_radial_fringe_points(
                pcd,
                k_neighbors=args.fringe_k_neighbors,
                margin=args.fringe_margin,
                reference_quantile=args.fringe_reference_quantile,
            )
        if visualize:
            o3d.visualization.draw_geometries([pcd])

        if not args.disable_cluster_filter:
            pcd = pcd.voxel_down_sample(0.005)
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=args.cluster_eps,
                    min_points=args.cluster_min_points,
                    print_progress=True,
                )
            )
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                largest = labels == np.bincount(labels[labels >= 0]).argmax()
                pcd = pcd.select_by_index(np.where(largest)[0])

        if args.manual_remove:
            pcd = manually_remove_points(
                pcd,
                distance=args.manual_remove_distance,
                max_remove_ratio=args.manual_max_remove_ratio,
                left_clicking=args.manual_left_click,
            )

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        if visualize:
            o3d.visualization.draw_geometries([pcd])

        dsf = dsf_fit.optimize_shape(dsf_cfg, np.asarray(pcd.points).T)
        dsf_v = pv.PolyData(dsf.v.T)

        if visualize or args.show_result:
            plotter = pv.Plotter()
            dsf.render(plotter, opacity=0.3)
            plotter.add_mesh(
                dsf_v, color="black", point_size=10, render_points_as_spheres=True
            )
            points = np.asarray(pcd.points)
            cloud = pv.PolyData(points)
            plotter.add_points(cloud)
            plotter.show()
        pcd_points = np.asarray(pcd.points)
        mass, inertia, center = compute_hull_inertial_parameters(pcd_points)
        exporter = WavefrontExporter(
            pcd_file.parent / "dsf.obj",
            {
                "mass": mass,
                "inertia": inertia,
            },
        )

        vertex = np.array(dsf.v.T) - center
        exporter.add_convex(
            vertices=vertex, sharpness=int(np.ceil(np.c_[dsf.p])), mu=args.mu
        )

        pcd.translate(-center)
        if not args.not_save_refined:
            o3d.io.write_point_cloud(pcd_file.parent / "merged_refined.pcd", pcd)


if __name__ == "__main__":

    if sys.gettrace():
        sys.argv = [__file__, "data/stone_pcd/pcd_251024/stone01"]

    main()
