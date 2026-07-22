import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def remove_radial_fringe_points(
    pcd: o3d.geometry.PointCloud,
    k_neighbors: int = 80,
    margin: float = 0.04,
    center_quantile: float = 0.5,
    reference_quantile: float = 0.65,
) -> o3d.geometry.PointCloud:
    """
    Trim outward fringe bands by comparing each point with nearby directions.

    Merged in-hand scans are roughly a shell around the stone. Registration and
    gripper-removal errors often create outward streaks in the same scan-line
    directions. This filter removes only points that are farther from the cloud
    center than the local radial envelope, instead of using point density.
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    if n_points == 0:
        return pcd

    center = np.quantile(points, center_quantile, axis=0)
    vectors = points - center
    radii = np.linalg.norm(vectors, axis=1)
    valid = radii > 1e-9
    if valid.sum() < 3:
        return pcd

    directions = np.zeros_like(vectors)
    directions[valid] = vectors[valid] / radii[valid, None]

    valid_idx = np.where(valid)[0]
    k = min(k_neighbors, len(valid_idx))
    tree = cKDTree(directions[valid_idx])
    _, neighbor_local_idx = tree.query(directions[valid_idx], k=k)
    if k == 1:
        neighbor_local_idx = neighbor_local_idx[:, None]

    neighbor_radii = radii[valid_idx][neighbor_local_idx]
    local_reference = np.quantile(neighbor_radii, reference_quantile, axis=1)
    remove_valid = radii[valid_idx] > local_reference + margin

    keep = np.ones(n_points, dtype=bool)
    keep[valid_idx[remove_valid]] = False
    return pcd.select_by_index(np.where(keep)[0])


def remove_detached_clusters(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.01,
    eps: float = 0.04,
    min_points: int = 6,
    attach_distance: float = 0.08,
    min_attached_size: int = 80,
    keep_ratio: float = 0.10,
    verbose: bool = False,
) -> o3d.geometry.PointCloud:
    """
    Remove small disconnected islands while preserving the main stone component.

    DBSCAN runs on a downsampled copy for speed. The retained downsampled
    components are then expanded by nearest-neighbor lookup to keep the original
    point density.
    """
    if len(pcd.points) == 0:
        return pcd

    pcd_down = pcd.voxel_down_sample(voxel) if voxel > 0 else pcd
    labels = np.array(
        pcd_down.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return pcd

    down_points = np.asarray(pcd_down.points)
    counts = np.bincount(valid_labels)
    largest_label = counts.argmax()
    largest_size = counts[largest_label]
    largest_points = down_points[labels == largest_label]
    largest_tree = cKDTree(largest_points)

    keep_labels = [largest_label]
    removed_labels = []
    for label, count in enumerate(counts):
        if label == largest_label:
            continue
        component_points = down_points[labels == label]
        min_dist_to_largest = largest_tree.query(component_points, k=1)[0].min()
        keep_attached_surface = (
            min_dist_to_largest <= attach_distance and count >= min_attached_size
        )
        keep_large_component = count >= largest_size * keep_ratio
        if keep_attached_surface or keep_large_component:
            keep_labels.append(label)
        else:
            removed_labels.append((label, int(count), float(min_dist_to_largest)))

    if verbose:
        print(
            f"[detached] components={len(counts)}, kept={len(keep_labels)}, "
            f"removed={len(removed_labels)}, largest={int(largest_size)}"
        )
        for label, count, min_dist in removed_labels[:10]:
            print(
                f"[detached] remove label={label}, size={count}, "
                f"dist_to_largest={min_dist:.4f}"
            )

    keep_down_points = down_points[np.isin(labels, keep_labels)]
    if len(keep_down_points) == 0:
        return pcd

    points = np.asarray(pcd.points)
    tree = cKDTree(keep_down_points)
    dists, _ = tree.query(points, k=1)
    keep = dists <= max(voxel * 1.75, eps * 0.25)
    return pcd.select_by_index(np.where(keep)[0])


def compute_hull_inertial_parameters(
    points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate mass properties by filling the convex hull of the point cloud.

    Density is 1, so mass equals hull volume. The returned inertia is about the
    returned center of mass and expressed in the point-cloud frame axes.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points to compute hull inertial parameters.")

    hull = ConvexHull(points)
    ref = points.mean(axis=0)

    tetrahedra = []
    total_volume = 0.0
    first_moment = np.zeros(3)
    for simplex in hull.simplices:
        tetra = np.vstack([ref, points[simplex]])
        volume = abs(np.linalg.det(tetra[1:] - tetra[0])) / 6.0
        if volume <= 0.0:
            continue
        tetrahedra.append((tetra, volume))
        total_volume += volume
        first_moment += volume * tetra.mean(axis=0)

    if total_volume <= 0.0:
        raise ValueError("Point-cloud convex hull has zero volume.")

    center = first_moment / total_volume
    second_moment = np.zeros((3, 3))
    for tetra, volume in tetrahedra:
        second_moment += _tetra_second_moment(tetra - center, volume)

    trace_second_moment = np.trace(second_moment)
    inertia = trace_second_moment * np.eye(3) - second_moment
    inertia = 0.5 * (inertia + inertia.T)
    return float(total_volume), inertia, center


def _tetra_second_moment(vertices: np.ndarray, volume: float) -> np.ndarray:
    vertex_sum = vertices.sum(axis=0)
    vertex_outer_sum = np.einsum("ni,nj->ij", vertices, vertices)
    return volume / 20.0 * (np.outer(vertex_sum, vertex_sum) + vertex_outer_sum)
