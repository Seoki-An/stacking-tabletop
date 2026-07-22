import numpy as np
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import trimesh
import open3d as o3d
from scipy.spatial import cKDTree


def box_crop_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    width: List[float] = [-4.0, 4.0],
    length: List[float] = [-4.0, 4.0],
    height: List[float] = [-4.0, 4.0],
    voxel: float = 0.003,
    cluster: bool = True,
) -> o3d.geometry.PointCloud:
    if voxel > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel)
    points = np.asarray(pcd.points)

    idx = np.where(
        (points[:, 0] > width[0])
        & (points[:, 0] < width[1])
        & (points[:, 1] > length[0])
        & (points[:, 1] < length[1])
        & (points[:, 2] > height[0])
        & (points[:, 2] < height[1])
    )[0]

    pcd = pcd.select_by_index(idx)

    if cluster:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True)
        )
        largest = labels == np.bincount(labels[labels >= 0]).argmax()
        pcd = pcd.select_by_index(np.where(largest)[0])

    return pcd


def get_nearby_points(
    pcd_full: o3d.geometry.PointCloud, pcd_subset: o3d.geometry.PointCloud, radius=0.05
) -> o3d.geometry.PointCloud:
    points_subset = np.asarray(pcd_subset.points)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)

    indices_set = set()
    for p in points_subset:
        _, idx, _ = pcd_tree.search_radius_vector_3d(p, radius)
        indices_set.update(idx)

    indices = np.array(list(indices_set))

    pcd_near = pcd_full.select_by_index(indices)
    pcd_near.paint_uniform_color([1, 0, 0])

    pcd_far = pcd_full.select_by_index(indices, invert=True)
    pcd_far.paint_uniform_color([0, 1, 0])

    # o3d.visualization.draw_geometries([pcd_near, pcd_far], point_show_normal=True)

    return pcd_near, pcd_far


def trimesh_to_open3d(trimesh_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    if trimesh_mesh.vertex_normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(trimesh_mesh.vertex_normals)
    else:
        mesh.compute_vertex_normals()
    return mesh


def open3d_to_trimesh(open3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(open3d_mesh.vertices),
        faces=np.asarray(open3d_mesh.triangles),
        process=False,
    )


def remove_points_inside_aabb(
    pcd: o3d.geometry.PointCloud, aabb: o3d.geometry.AxisAlignedBoundingBox
) -> o3d.geometry.PointCloud:
    indices = aabb.get_point_indices_within_bounding_box(pcd.points)
    filtered = pcd.select_by_index(indices, invert=True)
    return filtered


def remove_points_inside_sdf(
    pcd: o3d.geometry.PointCloud, voxelized_sdf: o3d.geometry.VoxelGrid
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    voxel_centers = voxelized_sdf.points
    kdtree = cKDTree(voxel_centers)
    dist, idx = kdtree.query(pts)
    threshold = 0.10
    mask = dist > threshold
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[mask])

    return filtered


def remove_points_from_points(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    threshold: float,
    cluster: bool = False,
) -> o3d.geometry.PointCloud:

    target_points = np.asarray(target_pcd.points)
    source_points = np.asarray(source_pcd.points)

    if len(target_points) == 0 or len(source_points) == 0:
        return copy.deepcopy(target_pcd)

    tree = cKDTree(source_points)
    dists, _ = tree.query(target_points, k=1)
    mask = dists >= threshold

    filtered_points = target_points[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    if cluster:
        labels = np.array(
            filtered_pcd.cluster_dbscan(eps=0.1, min_points=20, print_progress=False)
        )
        largest = labels == np.bincount(labels[labels >= 0]).argmax()
        filtered_pcd = filtered_pcd.select_by_index(np.where(largest)[0])

    return filtered_pcd


def select_points_near_points(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    threshold: float,
) -> o3d.geometry.PointCloud:
    target_points = np.asarray(target_pcd.points)
    source_points = np.asarray(source_pcd.points)

    if len(target_points) == 0 or len(source_points) == 0:
        return o3d.geometry.PointCloud()

    tree = cKDTree(source_points)
    dists, _ = tree.query(target_points, k=1)
    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(target_points[dists <= threshold])
    return selected_pcd


def multiscale_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_trans: np.ndarray = np.eye(4),
    voxel_sizes: List[float] = [0.1, 0.05, 0.02],
    max_iters: List[int] = [50, 30, 14],
    method="point_to_point",
    max_correspondence_distance_scale: float = 1.5,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, int]]]:
    """
    Perform multi-scale ICP from coarse to fine.

    Args:
        source_pcd: o3d.geometry.PointCloud
        target_pcd: o3d.geometry.PointCloud
        init_trans: (4x4) initial transformation
        voxel_sizes: list of voxel sizes for each scale
        max_iters: list of ICP iteration counts per scale
        method: "point_to_plane" or "point_to_point"
        max_correspondence_distance_scale: multiplier for each voxel size; the
            max correspondence distance at a scale is voxel * scale.
    Returns:
        final_trans: 4x4 transformation matrix
        history: list of (scale, fitness, rmse, correspondence_count)
    """
    assert len(voxel_sizes) == len(max_iters)

    current_trans = init_trans.copy()
    history = []

    for voxel, iters in zip(voxel_sizes, max_iters):

        src_down = source_pcd.voxel_down_sample(voxel)
        tgt_down = target_pcd.voxel_down_sample(voxel)

        radius_normal = voxel * 2.0
        src_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        tgt_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        if method == "point_to_plane":
            estimation = (
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        elif method == "point_to_point":
            estimation = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        threshold = voxel * float(max_correspondence_distance_scale)

        reg = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            max_correspondence_distance=threshold,
            init=current_trans,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters
            ),
        )
        current_trans = reg.transformation
        # info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        #     src_down,
        #     tgt_down,
        #     threshold,
        #     current_trans,
        # )

        history.append(
            (voxel, reg.fitness, reg.inlier_rmse, len(reg.correspondence_set))
        )

    return current_trans, history


def refine_pose_graph(
    pcds: List[o3d.geometry.PointCloud],
    voxel: float = 0.01,
    max_correspondence_distance: float = 0.02,
    fitness_threshold: float = 0.5,
    sequential_fitness_threshold: float = 0.15,
    reference_index: int = None,
    n_workers: int = 4,
    registration_pcds: List[o3d.geometry.PointCloud] = None,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Refine alignment of approximately-aligned PCDs via pose-graph optimization.

    Inputs must already be roughly in a common frame (e.g., gripper-canonical
    after per-frame ICP). Edges can be estimated from ``registration_pcds`` while
    the returned transforms are applied to ``pcds``. This is useful for in-hand
    scanning: the synthesized gripper is the stable object for graph edges, while
    the lidar cloud contains partially-overlapping stone observations.

    Returns a list of 4x4 transforms; ``pcds[i].transform(result[i])`` brings
    each cloud into the optimized common frame.
    """
    n = len(pcds)
    if n < 2:
        return [np.eye(4) for _ in pcds]
    if registration_pcds is None:
        registration_pcds = pcds
    if len(registration_pcds) != n:
        raise ValueError("registration_pcds must have the same length as pcds")

    pcds_down = []
    for pcd in registration_pcds:
        pd = pcd.voxel_down_sample(voxel)
        pd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)
        )
        pcds_down.append(pd)

    if reference_index is None:
        reference_index = max(range(n), key=lambda i: len(pcds_down[i].points))

    edge_pairs = [(i, i + 1, False) for i in range(n - 1)]
    edge_pairs += [
        (i, reference_index, True) for i in range(n) if i != reference_index
    ]

    voxel_sizes = [max(voxel * 5, 0.05), max(voxel * 2.5, 0.025), voxel]
    max_iters = [60, 35, 20]

    def register_pair(pair):
        i, j, uncertain = pair
        src, tgt = pcds_down[i], pcds_down[j]
        if len(src.points) < 50 or len(tgt.points) < 50:
            return None
        T, history = multiscale_icp(
            src,
            tgt,
            init_trans=np.eye(4),
            voxel_sizes=voxel_sizes,
            max_iters=max_iters,
            method="point_to_plane",
        )
        _, fitness, rmse, n_correspondence = history[-1]
        edge_fitness_threshold = (
            fitness_threshold if uncertain else sequential_fitness_threshold
        )
        if fitness < edge_fitness_threshold or n_correspondence < 30:
            return None
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src, tgt, max_correspondence_distance, T
        )
        return (i, j, np.asarray(T), info, uncertain, fitness, rmse, n_correspondence)

    if n_workers and n_workers > 1 and len(edge_pairs) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(register_pair, edge_pairs))
    else:
        results = [register_pair(p) for p in edge_pairs]

    pose_graph = o3d.pipelines.registration.PoseGraph()
    for _ in range(n):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    for r in results:
        if r is None:
            continue
        i, j, T, info, uncertain, fitness, rmse, n_correspondence = r
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, j, T, info, uncertain=uncertain
            )
        )
        if verbose:
            edge_type = "loop" if uncertain else "seq"
            print(
                f"[pose_graph] {edge_type} {i}->{j}: "
                f"fitness={fitness:.3f}, rmse={rmse:.4f}, corr={n_correspondence}"
            )

    if not pose_graph.edges:
        if verbose:
            print("[pose_graph] no accepted edges; returning identity refinements")
        return [np.eye(4) for _ in pcds]

    components = _pose_graph_components(n, pose_graph.edges)
    if verbose:
        print(
            f"[pose_graph] accepted_edges={len(pose_graph.edges)}, "
            f"components={len(components)}, sizes={[len(c) for c in components]}"
        )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.0,
        reference_node=reference_index,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    return [np.asarray(pose_graph.nodes[i].pose).copy() for i in range(n)]


def _pose_graph_components(n_nodes, edges):
    neighbors = [set() for _ in range(n_nodes)]
    for edge in edges:
        neighbors[edge.source_node_id].add(edge.target_node_id)
        neighbors[edge.target_node_id].add(edge.source_node_id)

    seen = set()
    components = []
    for start in range(n_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in neighbors[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(component)
    return components


def pose_graph_icp(
    pcds: List[o3d.geometry.PointCloud], threshold: float = 0.01
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
    pose_graph = o3d.pipelines.registration.PoseGraph()

    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    current_pose = np.eye(4)

    for i in range(len(pcds) - 1):
        src = pcds[i]
        tgt = pcds[i + 1]

        T, _ = multiscale_icp(
            src,
            tgt,
            init_trans=np.eye(4),
            voxel_sizes=[0.05, 0.03, 0.02, 0.01],
            max_iters=[100, 50, 30, 14],
        )

        current_pose = current_pose @ T

        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(current_pose))
        )
        src_tmp = copy.deepcopy(src)
        src_tmp.transform(T)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_tmp,
            tgt,
            threshold,
            T,
        )

        edge = o3d.pipelines.registration.PoseGraphEdge(
            i, i + 1, T, info, uncertain=True
        )
        pose_graph.edges.append(edge)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=threshold * 1.2,
        edge_prune_threshold=0.10,
        preference_loop_closure=2.0,
        reference_node=0,
    )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    pcds_optimized = []
    pose_optimized = []
    for i, pcd in enumerate(pcds):
        T = pose_graph.nodes[i].pose
        pcd_opt = copy.deepcopy(pcd)
        pcd_opt.transform(T)
        pcds_optimized.append(pcd_opt)
        pose_optimized.append(T)

    return pcds_optimized, pose_optimized
