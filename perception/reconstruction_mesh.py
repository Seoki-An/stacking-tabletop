import argparse
import pathlib

import open3d as o3d
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def pcd_to_refined_mesh(
    pcd: o3d.geometry.PointCloud,
    method: str = "radial",
    reconstruction_voxel: float = 0.02,
    alpha: float = 0.10,
    bpa_radius_scales: tuple[float, ...] = (1.5, 2.5, 4.0),
    radial_subdivisions: int = 5,
    radial_k_neighbors: int = 80,
    radial_radius_quantile: float = 0.8,
    radial_radius_padding: float = 0.0,
    radial_center_quantile: float = 0.5,
    radial_empty_angle_deg: float = 25.0,
    radial_empty_radius_quantile: float = 0.5,
    radial_support_angle_deg: float = 0.0,
    radial_min_support_points: int = 0,
    radial_tighten_angle_deg: float = 0.0,
    radial_tighten_quantile: float = 0.9,
    radial_tighten_blend: float = 1.0,
    radial_max_radius_quantile: float = 0.99,
    radial_radius_smooth_iterations: int = 3,
    radial_radius_smooth_weight: float = 0.3,
    normal_radius: float = 0.08,
    normal_max_nn: int = 40,
    poisson_depth: int = 12,
    density_quantile: float = 0.02,
    smooth_iterations: int = 2,
    simplify_voxel: float = 0.0,
    keep_largest_component: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a triangle mesh from a cleaned point cloud.

    The input point cloud is expected to already be filtered and centered in the
    same frame as the saved refined PCD/DSF model.
    """
    if len(pcd.points) < 4:
        raise ValueError("Need at least 4 points to reconstruct a mesh.")

    pcd_for_mesh = o3d.geometry.PointCloud(pcd)
    if reconstruction_voxel > 0.0:
        pcd_for_mesh = pcd_for_mesh.voxel_down_sample(reconstruction_voxel)

    if len(pcd_for_mesh.points) < 4:
        raise ValueError("Need at least 4 downsampled points to reconstruct a mesh.")

    method = method.lower()
    if method == "radial":
        mesh = create_radial_ball_mesh(
            pcd_for_mesh,
            subdivisions=radial_subdivisions,
            k_neighbors=radial_k_neighbors,
            radius_quantile=radial_radius_quantile,
            radius_padding=radial_radius_padding,
            center_quantile=radial_center_quantile,
            empty_angle_deg=radial_empty_angle_deg,
            empty_radius_quantile=radial_empty_radius_quantile,
            support_angle_deg=radial_support_angle_deg,
            min_support_points=radial_min_support_points,
            tighten_angle_deg=radial_tighten_angle_deg,
            tighten_quantile=radial_tighten_quantile,
            tighten_blend=radial_tighten_blend,
            max_radius_quantile=radial_max_radius_quantile,
            smooth_iterations=radial_radius_smooth_iterations,
            smooth_weight=radial_radius_smooth_weight,
        )
    elif method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd_for_mesh,
            alpha,
        )
    elif method == "bpa":
        if not pcd_for_mesh.has_normals():
            pcd_for_mesh.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius, max_nn=normal_max_nn
                )
            )
        pcd_for_mesh.orient_normals_consistent_tangent_plane(normal_max_nn)
        nn_distances = np.asarray(pcd_for_mesh.compute_nearest_neighbor_distance())
        radius_base = float(np.median(nn_distances))
        radii = o3d.utility.DoubleVector(
            [radius_base * scale for scale in bpa_radius_scales]
        )
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_for_mesh,
            radii,
        )
    elif method == "poisson":
        if not pcd_for_mesh.has_normals():
            pcd_for_mesh.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius, max_nn=normal_max_nn
                )
            )
        pcd_for_mesh.orient_normals_consistent_tangent_plane(normal_max_nn)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_for_mesh,
            depth=poisson_depth,
        )

        if len(densities) > 0 and density_quantile > 0.0:
            densities = np.asarray(densities)
            keep_threshold = np.quantile(densities, density_quantile)
            mesh.remove_vertices_by_mask(densities < keep_threshold)

        bbox = pcd_for_mesh.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.03, bbox.get_center())
        mesh = mesh.crop(bbox)
    else:
        raise ValueError(
            f"Unsupported mesh reconstruction method: {method}. "
            "Choose one of: radial, alpha, bpa, poisson."
        )

    mesh = clean_mesh(mesh, keep_largest_component=keep_largest_component)

    if simplify_voxel > 0.0 and len(mesh.triangles) > 0:
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=simplify_voxel,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        mesh = clean_mesh(mesh, keep_largest_component=keep_largest_component)

    if smooth_iterations > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        mesh = clean_mesh(mesh, keep_largest_component=keep_largest_component)

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def create_radial_ball_mesh(
    pcd: o3d.geometry.PointCloud,
    subdivisions: int = 4,
    k_neighbors: int = 80,
    radius_quantile: float = 0.95,
    radius_padding: float = 0.0,
    center_quantile: float = 0.5,
    empty_angle_deg: float = 35.0,
    empty_radius_quantile: float = 0.70,
    support_angle_deg: float = 0.0,
    min_support_points: int = 0,
    tighten_angle_deg: float = 0.0,
    tighten_quantile: float = 0.9,
    tighten_blend: float = 1.0,
    max_radius_quantile: float = 0.99,
    smooth_iterations: int = 3,
    smooth_weight: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """
    Build a closed star-convex mesh from a point cloud.

    This encodes the prior that the observed points are samples from the outside
    of one stone-like solid: every direction from the center receives one outer
    radius, and missing directions are filled from nearby observed directions.
    """
    points = np.asarray(pcd.points)
    if len(points) < 4:
        raise ValueError("Need at least 4 points to create a radial mesh.")

    center = np.quantile(points, center_quantile, axis=0)
    vectors = points - center
    radii = np.linalg.norm(vectors, axis=1)
    valid = radii > 1e-9
    if valid.sum() < 4:
        raise ValueError("Point cloud is too close to its center to create a mesh.")

    point_dirs = vectors[valid] / radii[valid, None]
    point_radii = radii[valid]

    sphere = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
    if subdivisions > 0:
        sphere = sphere.subdivide_midpoint(number_of_iterations=subdivisions)
    sphere_vertices = np.asarray(sphere.vertices)
    sphere_dirs = sphere_vertices / np.linalg.norm(sphere_vertices, axis=1)[:, None]

    k = min(max(1, k_neighbors), len(point_dirs))
    tree = cKDTree(point_dirs)
    neighbor_dist, neighbor_idx = tree.query(sphere_dirs, k=k)
    if k == 1:
        neighbor_dist = neighbor_dist[:, None]
        neighbor_idx = neighbor_idx[:, None]

    neighbor_radii = point_radii[neighbor_idx]
    surface_radii = np.quantile(neighbor_radii, radius_quantile, axis=1)

    nearest_angles = 2.0 * np.arcsin(np.clip(neighbor_dist[:, 0] * 0.5, 0.0, 1.0))
    empty_angle = np.deg2rad(empty_angle_deg)
    if empty_angle > 0.0:
        fill_radius = np.quantile(point_radii, empty_radius_quantile)
        empty_weight = np.clip(nearest_angles / empty_angle, 0.0, 1.0)
        surface_radii = (1.0 - empty_weight) * surface_radii + empty_weight * min(
            fill_radius,
            np.median(surface_radii),
        )

    support_angle = np.deg2rad(support_angle_deg)
    if support_angle > 0.0 and min_support_points > 0:
        support_chord = 2.0 * np.sin(support_angle * 0.5)
        support_counts = np.fromiter(
            (
                len(indices)
                for indices in tree.query_ball_point(sphere_dirs, r=support_chord)
            ),
            dtype=np.int32,
            count=len(sphere_dirs),
        )
        fill_radius = np.quantile(point_radii, empty_radius_quantile)
        fill_radius = min(fill_radius, np.median(surface_radii))
        support_weight = np.clip(
            (min_support_points - support_counts) / max(min_support_points, 1),
            0.0,
            1.0,
        )
        surface_radii = (
            1.0 - support_weight
        ) * surface_radii + support_weight * fill_radius

    tighten_angle = np.deg2rad(tighten_angle_deg)
    if tighten_angle > 0.0 and tighten_blend > 0.0:
        tighten_chord = 2.0 * np.sin(tighten_angle * 0.5)
        tighten_radii = surface_radii.copy()
        for idx, neighbor_ids in enumerate(
            tree.query_ball_point(sphere_dirs, r=tighten_chord)
        ):
            if neighbor_ids:
                tighten_radii[idx] = np.quantile(
                    point_radii[neighbor_ids],
                    np.clip(tighten_quantile, 0.0, 1.0),
                )
        tighten_blend = float(np.clip(tighten_blend, 0.0, 1.0))
        surface_radii = np.minimum(
            surface_radii,
            (1.0 - tighten_blend) * surface_radii + tighten_blend * tighten_radii,
        )

    if max_radius_quantile > 0.0:
        max_radius = np.quantile(point_radii, max_radius_quantile)
        surface_radii = np.minimum(surface_radii, max_radius)

    if smooth_iterations > 0 and smooth_weight > 0.0:
        surface_radii = smooth_mesh_vertex_values(
            surface_radii,
            np.asarray(sphere.triangles),
            iterations=smooth_iterations,
            weight=smooth_weight,
        )

    surface_radii += radius_padding
    surface_radii = np.maximum(surface_radii, 1e-6)

    vertices = center + sphere_dirs * surface_radii[:, None]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = sphere.triangles
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def smooth_mesh_vertex_values(
    values: np.ndarray,
    triangles: np.ndarray,
    iterations: int = 3,
    weight: float = 0.5,
) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    weight = float(np.clip(weight, 0.0, 1.0))
    if iterations <= 0 or weight <= 0.0 or len(triangles) == 0:
        return values

    neighbors = [set() for _ in range(len(values))]
    for tri in triangles:
        a, b, c = tri
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    for _ in range(iterations):
        smoothed = values.copy()
        for idx, nbrs in enumerate(neighbors):
            if nbrs:
                smoothed[idx] = np.mean(values[list(nbrs)])
        values = (1.0 - weight) * values + weight * smoothed

    return values


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
    keep_largest_component: bool = True,
) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if keep_largest_component and len(mesh.triangles) > 0:
        triangle_labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if len(counts) > 1:
            largest_label = int(counts.argmax())
            remove_mask = np.asarray(triangle_labels) != largest_label
            mesh.remove_triangles_by_mask(remove_mask)
            mesh.remove_unreferenced_vertices()

    return mesh


def visualize_mesh_with_pcd(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
    mesh_opacity: float = 0.35,
) -> None:
    points = np.asarray(pcd.points)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    plotter = pv.Plotter()
    if len(points) > 0:
        cloud = pv.PolyData(points)
        cloud["z"] = points[:, 2]
        plotter.add_points(
            cloud,
            scalars="z",
            point_size=4,
            render_points_as_spheres=False,
            name="pcd",
        )

    if len(vertices) > 0 and len(triangles) > 0:
        faces = np.c_[np.full(len(triangles), 3), triangles].astype(np.int64)
        poly = pv.PolyData(vertices, faces)
        plotter.add_mesh(
            poly,
            color="lightsteelblue",
            opacity=mesh_opacity,
            show_edges=True,
            edge_color="black",
            name="mesh",
        )

    plotter.show()


def find_pcd_paths(root_path: pathlib.Path, input_name: str) -> list[pathlib.Path]:
    if root_path.is_file():
        return [root_path]
    if not root_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root_path}")
    return sorted(root_path.rglob(input_name))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create refined mesh models from merged_refined.pcd files generated "
            "by perception/reconstruction.py."
        )
    )
    parser.add_argument(
        "root_path",
        help="A merged_refined.pcd file or a directory containing refined PCDs.",
    )
    parser.add_argument(
        "--input_name",
        default="merged_refined.pcd",
        help="PCD filename to search for when root_path is a directory.",
    )
    parser.add_argument(
        "--output_name",
        default="mesh_refined.ply",
        help="Mesh filename written next to each input PCD.",
    )
    parser.add_argument(
        "--show_result",
        action="store_true",
        help="Visualize the input PCD with the reconstructed mesh overlaid.",
    )
    parser.add_argument(
        "--mesh_method",
        default="radial",
        choices=["radial", "alpha", "bpa", "poisson"],
        help=(
            "Surface reconstruction method. Radial is the default for "
            "watertight stone-like solids."
        ),
    )
    parser.add_argument(
        "--mesh_reconstruction_voxel",
        default=0.02,
        type=float,
        help="Voxel size used to downsample the PCD before mesh reconstruction.",
    )
    parser.add_argument(
        "--mesh_alpha",
        default=0.10,
        type=float,
        help="Alpha radius for alpha-shape mesh reconstruction.",
    )
    parser.add_argument(
        "--mesh_bpa_radius_scales",
        default="1.5,2.5,4.0",
        help="Comma-separated nearest-neighbor radius scales for ball pivoting.",
    )
    parser.add_argument(
        "--mesh_radial_subdivisions",
        default=4,
        type=int,
        help="Icosphere subdivision level for radial watertight mesh reconstruction.",
    )
    parser.add_argument(
        "--mesh_radial_k_neighbors",
        default=120,
        type=int,
        help="Directional neighbors used to estimate each radial surface vertex.",
    )
    parser.add_argument(
        "--mesh_radial_radius_quantile",
        default=0.7,
        type=float,
        help="Neighbor radius quantile used as the outer stone surface.",
    )
    parser.add_argument(
        "--mesh_radial_radius_padding",
        default=0.0,
        type=float,
        help="Extra outward radius added to radial mesh vertices.",
    )
    parser.add_argument(
        "--mesh_radial_center_quantile",
        default=0.5,
        type=float,
        help="Coordinate-wise point quantile used as the radial mesh center.",
    )
    parser.add_argument(
        "--mesh_radial_empty_angle_deg",
        default=25.0,
        type=float,
        help="Directions farther than this from observed PCD shrink to fill radius.",
    )
    parser.add_argument(
        "--mesh_radial_empty_radius_quantile",
        default=0.75,
        type=float,
        help="Global radius quantile used to fill poorly observed directions.",
    )
    parser.add_argument(
        "--mesh_radial_support_angle_deg",
        default=8,
        type=float,
        help=(
            "Treat a radial direction as empty when too few PCD directions lie "
            "within this angular radius. Disabled when 0."
        ),
    )
    parser.add_argument(
        "--mesh_radial_min_support_points",
        default=0,
        type=int,
        help=(
            "Minimum directional support count required inside "
            "--mesh_radial_support_angle_deg. Disabled when 0."
        ),
    )
    parser.add_argument(
        "--mesh_radial_tighten_angle_deg",
        default=5.0,
        type=float,
        help=(
            "Clamp each radial vertex toward nearby PCD radii inside this "
            "angular radius. Disabled when 0."
        ),
    )
    parser.add_argument(
        "--mesh_radial_tighten_quantile",
        default=0.9,
        type=float,
        help="Local PCD radius quantile used by --mesh_radial_tighten_angle_deg.",
    )
    parser.add_argument(
        "--mesh_radial_tighten_blend",
        default=1.0,
        type=float,
        help="Blend toward the local tighten quantile, from 0 to 1.",
    )
    parser.add_argument(
        "--mesh_radial_max_radius_quantile",
        default=0.99,
        type=float,
        help="Global radius quantile used as an upper cap against radial bulges.",
    )
    parser.add_argument(
        "--mesh_radial_radius_smooth_iterations",
        default=3,
        type=int,
        help="Neighbor smoothing iterations for radial mesh vertex radii.",
    )
    parser.add_argument(
        "--mesh_radial_radius_smooth_weight",
        default=0.1,
        type=float,
        help="Blend weight for each radial radius smoothing iteration.",
    )
    parser.add_argument(
        "--mesh_normal_radius",
        default=0.05,
        type=float,
        help="Neighbor radius used for mesh normal estimation.",
    )
    parser.add_argument(
        "--mesh_normal_max_nn",
        default=40,
        type=int,
        help="Maximum neighbors used for mesh normal estimation/orientation.",
    )
    parser.add_argument(
        "--mesh_poisson_depth",
        default=12,
        type=int,
        help="Poisson reconstruction depth. Higher values preserve more detail.",
    )
    parser.add_argument(
        "--mesh_density_quantile",
        default=0.02,
        type=float,
        help="Remove mesh vertices below this Poisson density quantile.",
    )
    parser.add_argument(
        "--mesh_smooth_iterations",
        default=2,
        type=int,
        help="Taubin smoothing iterations applied after mesh cleanup.",
    )
    parser.add_argument(
        "--mesh_simplify_voxel",
        default=0.0,
        type=float,
        help="Optional vertex-clustering voxel size for mesh simplification.",
    )
    parser.add_argument(
        "--mesh_visualization_opacity",
        default=0.35,
        type=float,
        help="Opacity of the reconstructed mesh in --show_result visualization.",
    )
    parser.add_argument(
        "--keep_mesh_components",
        action="store_true",
        help="Keep all connected mesh components instead of only the largest one.",
    )
    args = parser.parse_args()

    pcd_paths = find_pcd_paths(pathlib.Path(args.root_path), args.input_name)
    if not pcd_paths:
        print(f"No {args.input_name} files found under {args.root_path}")
        return

    bpa_radius_scales = tuple(
        float(scale.strip())
        for scale in args.mesh_bpa_radius_scales.split(",")
        if scale.strip()
    )

    for pcd_path in pcd_paths:
        print(f"Processing: {pcd_path}")
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        mesh = pcd_to_refined_mesh(
            pcd,
            method=args.mesh_method,
            reconstruction_voxel=args.mesh_reconstruction_voxel,
            alpha=args.mesh_alpha,
            bpa_radius_scales=bpa_radius_scales,
            radial_subdivisions=args.mesh_radial_subdivisions,
            radial_k_neighbors=args.mesh_radial_k_neighbors,
            radial_radius_quantile=args.mesh_radial_radius_quantile,
            radial_radius_padding=args.mesh_radial_radius_padding,
            radial_center_quantile=args.mesh_radial_center_quantile,
            radial_empty_angle_deg=args.mesh_radial_empty_angle_deg,
            radial_empty_radius_quantile=args.mesh_radial_empty_radius_quantile,
            radial_support_angle_deg=args.mesh_radial_support_angle_deg,
            radial_min_support_points=args.mesh_radial_min_support_points,
            radial_tighten_angle_deg=args.mesh_radial_tighten_angle_deg,
            radial_tighten_quantile=args.mesh_radial_tighten_quantile,
            radial_tighten_blend=args.mesh_radial_tighten_blend,
            radial_max_radius_quantile=args.mesh_radial_max_radius_quantile,
            radial_radius_smooth_iterations=(args.mesh_radial_radius_smooth_iterations),
            radial_radius_smooth_weight=args.mesh_radial_radius_smooth_weight,
            normal_radius=args.mesh_normal_radius,
            normal_max_nn=args.mesh_normal_max_nn,
            poisson_depth=args.mesh_poisson_depth,
            density_quantile=args.mesh_density_quantile,
            smooth_iterations=args.mesh_smooth_iterations,
            simplify_voxel=args.mesh_simplify_voxel,
            keep_largest_component=not args.keep_mesh_components,
        )

        mesh_path = pcd_path.parent / args.output_name
        if args.show_result:
            visualize_mesh_with_pcd(
                mesh,
                pcd,
                mesh_opacity=args.mesh_visualization_opacity,
            )
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(
            f"Saved refined mesh: {mesh_path} "
            f"({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)"
        )


if __name__ == "__main__":
    main()
