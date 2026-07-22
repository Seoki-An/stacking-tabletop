import os
import omegaconf
import argparse
import pickle
import copy
import csv
import numpy as np
import open3d as o3d
import open3d.visualization.gui as o3d_gui
import open3d.visualization.rendering as o3d_rendering
from scipy.spatial.transform import Rotation

import rclpy

from model import get_stone_model, get_excavator_model
from perception import box_crop_largest_cluster, multiscale_icp
from perception.reconstruction_dsf import manually_remove_points
from perception.utils.refine_pcd import (
    refine_pose_graph,
    remove_points_from_points,
    remove_points_inside_aabb,
)

from ros2.pose_node import PoseArrayPublisher

_O3D_APP_INITIALIZED = False


def _ensure_app():
    global _O3D_APP_INITIALIZED
    if not _O3D_APP_INITIALIZED:
        o3d_gui.Application.instance.initialize()
        _O3D_APP_INITIALIZED = True
    return o3d_gui.Application.instance


def _draw_with_labels(geometries, labels, window_name="open3d"):
    """Show geometries with floating 3D text labels via gui.SceneWidget.add_3d_label.

    labels: list of (position, text) tuples.
    """
    app = _ensure_app()
    win = app.create_window(window_name, 1280, 720)
    sw = o3d_gui.SceneWidget()
    sw.scene = o3d_rendering.Open3DScene(win.renderer)
    sw.scene.set_background([0.9, 0.9, 0.9, 1.0])
    sw.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 45000)
    sw.scene.scene.enable_sun_light(True)
    win.add_child(sw)

    mat = o3d_rendering.MaterialRecord()
    mat.shader = "defaultLit"
    for idx, geom in enumerate(geometries):
        sw.scene.add_geometry(f"geom_{idx}", geom, mat)
    for pos, text in labels:
        sw.add_3d_label(list(pos), text)

    bounds = sw.scene.bounding_box
    sw.setup_camera(60.0, bounds, bounds.get_center())
    app.run()


def normalize_pose_quaternion(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1).copy()
    if pose.shape != (7,):
        raise ValueError(
            f"pose must have 7 values [x,y,z,qx,qy,qz,qw], got shape {pose.shape}"
        )
    quat_norm = float(np.linalg.norm(pose[3:]))
    if quat_norm <= 1e-12:
        raise ValueError(f"pose quaternion is degenerate: {pose}")
    pose[3:] /= quat_norm
    return pose


def parse_pose_vector(value: str) -> np.ndarray:
    text = str(value).strip().replace(";", ",").replace("|", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return normalize_pose_quaternion([float(p) for p in parts])


def parse_stone_id_pose(value: str) -> tuple[int, np.ndarray]:
    """Parse 'STONE_ID:x,y,z,qx,qy,qz,qw' into (stone_id, pose_7vec)."""
    if ":" not in value:
        raise ValueError(
            f"--manual_init_poses must be STONE_ID:x,y,z,qx,qy,qz,qw, got {value!r}"
        )
    stone_text, pose_text = value.split(":", 1)
    return int(stone_text.strip()), parse_pose_vector(pose_text)


def pose_data_entry_to_pose(value) -> np.ndarray:
    if isinstance(value, dict):
        if "pose" in value:
            value = value["pose"]
        elif "pos" in value and "quat" in value:
            value = (value["pos"], value["quat"])
        elif "position" in value and "orientation" in value:
            value = (value["position"], value["orientation"])

    if isinstance(value, (tuple, list)) and len(value) == 2:
        pos, quat = value
        value = np.concatenate(
            [
                np.asarray(pos, dtype=np.float64).reshape(3),
                np.asarray(quat, dtype=np.float64).reshape(4),
            ]
        )
    return normalize_pose_quaternion(np.asarray(value, dtype=np.float64).reshape(-1))


def load_pose_data_initial_poses(path: str) -> dict[int, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pose data file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Pose data must be a dict keyed by stone id: {path}")

    poses = {}
    for stone_id, value in data.items():
        poses[int(stone_id)] = pose_data_entry_to_pose(value)
    return poses


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:] = Rotation.from_matrix(T[:3, :3]).as_quat()
    return pose


def _set_mesh_pose(mesh, base_vertices, base_normals, pose):
    R = Rotation.from_quat(pose[3:]).as_matrix()
    mesh.vertices = o3d.utility.Vector3dVector(base_vertices @ R.T + pose[:3])
    if base_normals.size:
        mesh.vertex_normals = o3d.utility.Vector3dVector(base_normals @ R.T)
    else:
        mesh.compute_vertex_normals()


def run_manual_init_gui(
    cluster_pcds: list,
    ids: list,
    stone_meshes: dict,
    existing_manual_poses: dict,
    translation_step: float = 0.05,
    rotation_step_deg: float = 5.0,
    target_stone_ids: set | None = None,
) -> dict:
    """Interactively adjust ICP initial poses for each stone.

    Shows stone meshes overlaid on cluster point clouds.
    Initial pose comes from existing_manual_poses[stone_id] when available,
    otherwise cluster centroid + identity rotation.

    Controls:
      N / P       : next / previous stone
      1-9         : select stone by index
      A / D       : translate -/+ X
      S / W       : translate -/+ Y
      F / R       : translate -/+ Z
      J / L       : yaw  -/+ (Z axis)
      I / K       : pitch +/- (Y axis)
      U / O       : roll  -/+ (X axis)
      M           : mark current pose as manual without moving
      Z           : reset selected stone to its original pose
      C or Enter  : confirm and close

    Returns dict {stone_id: np.ndarray(7)} for all stones that were edited.
    """
    editable = []
    for stone_id in ids:
        if stone_id not in stone_meshes:
            continue
        if target_stone_ids is not None and stone_id not in target_stone_ids:
            continue
        if stone_id in existing_manual_poses:
            pose = existing_manual_poses[stone_id].copy()
        else:
            # Use the cluster centroid for this stone as default translation
            idx = ids.index(stone_id)
            center = np.asarray(cluster_pcds[idx].points).mean(0)
            pose = np.array([*center, 0.0, 0.0, 0.0, 1.0])
        editable.append((stone_id, pose))

    if not editable:
        print("  manual init GUI: no stone meshes available; skipping.")
        return {}

    poses_by_stone = {sid: p.copy() for sid, p in editable}
    original_poses = {sid: p.copy() for sid, p in editable}
    dirty: set[int] = set()
    selected = {"idx": 0}

    # Build combined scene point cloud (gray)
    scene_pcd = o3d.geometry.PointCloud()
    for pcd in cluster_pcds:
        c = o3d.geometry.PointCloud(pcd)
        c.paint_uniform_color([0.5, 0.5, 0.5])
        scene_pcd += c

    geoms = [
        scene_pcd,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]

    mesh_state = {}
    for stone_id, pose in editable:
        mesh = copy.deepcopy(stone_meshes[stone_id])
        mesh.compute_vertex_normals()
        bv = np.asarray(mesh.vertices).copy()
        bn = np.asarray(mesh.vertex_normals).copy()
        _set_mesh_pose(mesh, bv, bn, pose)
        mesh_state[stone_id] = {"mesh": mesh, "base_vertices": bv, "base_normals": bn}
        geoms.append(mesh)

    def _stone_color(idx, stone_id):
        if idx == selected["idx"]:
            return [1.0, 0.45, 0.05, 1.0]  # orange = selected
        elif stone_id in dirty:
            return [0.10, 0.70, 0.25, 1.0]  # green  = edited
        return [0.10, 0.35, 1.00, 1.0]  # blue   = untouched

    def _print_pose(prefix="selected"):
        stone_id, _ = editable[selected["idx"]]
        pose = poses_by_stone[stone_id]
        print(f"  [{prefix}] stone {stone_id}: {np.round(pose, 5).tolist()}")

    print(
        "\nManual init GUI controls:\n"
        "  N/P or 1-9 : next/prev or indexed stone\n"
        "  A/D        : move -/+ X\n"
        "  S/W        : move -/+ Y\n"
        "  F/R        : move -/+ Z\n"
        "  J/L        : yaw  -/+ (Z)\n"
        "  I/K        : pitch +/- (Y)\n"
        "  U/O        : roll  -/+ (X)\n"
        "  M          : mark without moving\n"
        "  Z          : reset selected\n"
        "  C / Enter  : confirm\n"
        f"  step: translation={translation_step} m, rotation={rotation_step_deg} deg"
    )
    _print_pose()

    app = _ensure_app()
    win = app.create_window("Manual init pose adjustment", 1280, 720)
    sw = o3d_gui.SceneWidget()
    sw.scene = o3d_rendering.Open3DScene(win.renderer)
    sw.scene.set_background([0.15, 0.15, 0.15, 1.0])
    sw.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 45000)
    sw.scene.scene.enable_sun_light(True)
    win.add_child(sw)

    _default_mat = o3d_rendering.MaterialRecord()
    _default_mat.shader = "defaultLit"

    # Add static geometry (scene PCD + coordinate frame)
    sw.scene.add_geometry("__scene__", scene_pcd, _default_mat)
    sw.scene.add_geometry(
        "__frame__",
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
        _default_mat,
    )

    # Track which stone geometries are currently in the scene
    _in_scene: set[int] = set()

    def _geom_name(stone_id):
        return f"stone_{stone_id}"

    def _upsert_stone(idx, stone_id):
        name = _geom_name(stone_id)
        if stone_id in _in_scene:
            sw.scene.remove_geometry(name)
        mat = o3d_rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = _stone_color(idx, stone_id)
        sw.scene.add_geometry(name, mesh_state[stone_id]["mesh"], mat)
        _in_scene.add(stone_id)

    def _repaint():
        for idx, (stone_id, _) in enumerate(editable):
            _upsert_stone(idx, stone_id)

    def _refresh(stone_id):
        s = mesh_state[stone_id]
        _set_mesh_pose(
            s["mesh"], s["base_vertices"], s["base_normals"], poses_by_stone[stone_id]
        )
        idx = next(i for i, (sid, _) in enumerate(editable) if sid == stone_id)
        _upsert_stone(idx, stone_id)

    # Initial paint
    _repaint()
    bounds = sw.scene.bounding_box
    sw.setup_camera(60.0, bounds, bounds.get_center())

    ax_x = np.array([1.0, 0.0, 0.0])
    ax_y = np.array([0.0, 1.0, 0.0])
    ax_z = np.array([0.0, 0.0, 1.0])
    t = translation_step

    def on_key(event):
        if event.type != o3d_gui.KeyEvent.DOWN:
            return o3d_gui.SceneWidget.EventCallbackResult.IGNORED
        k = event.key
        stone_id, _ = editable[selected["idx"]]

        if k == ord("N"):
            selected["idx"] = (selected["idx"] + 1) % len(editable)
            _repaint()
            _print_pose()
        elif k == ord("P"):
            selected["idx"] = (selected["idx"] - 1) % len(editable)
            _repaint()
            _print_pose()
        elif k == ord("A"):
            poses_by_stone[stone_id][:3] += [-t, 0, 0]
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("moved")
        elif k == ord("D"):
            poses_by_stone[stone_id][:3] += [+t, 0, 0]
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("moved")
        elif k == ord("S"):
            poses_by_stone[stone_id][:3] += [0, -t, 0]
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("moved")
        elif k == ord("W"):
            poses_by_stone[stone_id][:3] += [0, +t, 0]
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("moved")
        elif k == ord("F"):
            poses_by_stone[stone_id][:3] += [0, 0, -t]
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("moved")
        elif k == ord("R"):
            poses_by_stone[stone_id][:3] += [0, 0, +t]
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("moved")
        elif k in (ord("J"), ord("L"), ord("I"), ord("K"), ord("U"), ord("O")):
            axis = {
                ord("J"): ax_z,
                ord("L"): ax_z,
                ord("I"): ax_y,
                ord("K"): ax_y,
                ord("U"): ax_x,
                ord("O"): ax_x,
            }[k]
            sign = +1.0 if k in (ord("L"), ord("I"), ord("O")) else -1.0
            pose = poses_by_stone[stone_id]
            angle = np.deg2rad(rotation_step_deg * sign)
            new_rot = Rotation.from_rotvec(axis * angle) * Rotation.from_quat(pose[3:])
            pose[3:] = new_rot.as_quat()
            pose[:] = normalize_pose_quaternion(pose)
            dirty.add(stone_id)
            _refresh(stone_id)
            _print_pose("rotated")
        elif k == ord("M"):
            dirty.add(stone_id)
            _repaint()
            _print_pose("marked")
        elif k == ord("Z"):
            poses_by_stone[stone_id] = original_poses[stone_id].copy()
            dirty.discard(stone_id)
            _refresh(stone_id)
            _print_pose("reset")
        elif k == ord("C") or k == 13:  # C or Enter
            app.quit()
        elif ord("1") <= k <= ord("9"):
            idx = k - ord("1")
            if idx < len(editable):
                selected["idx"] = idx
                _repaint()
                _print_pose()

        return o3d_gui.SceneWidget.EventCallbackResult.HANDLED

    sw.set_on_key(on_key)
    app.run()

    result = {stone_id: poses_by_stone[stone_id].copy() for stone_id in dirty}
    if result:
        print(
            "  manual GUI poses set for stones: "
            + ", ".join(str(s) for s in sorted(result))
        )
    else:
        print("  manual GUI: no poses edited.")
    return result


POSEID_MODE = 0
GLOBAL_RANSAC_THRESHOLD = 0.15
LOCAL_RANSAC_THRESHOLD = 0.05
GLOBAL_CLUSTER_THRESHOLD = 0.10
LOCAL_CLUSTER_THRESHOLD = 0.10
LIDAR_LINK_NAMES = ("lidar1_link", "lidar2_link", "lidar3_link")
Z_THRESHOLD = -0.10
# Maps local X→+X (no mirror), local Y→+Z (upright), local Z→+Y (faces +Y camera).
# det=-1 (improper); make_text_mesh flips triangle winding afterwards to restore
# face normals toward the camera.
TEXT_LABEL_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
)


def get_link_transform(model, link_name):
    transform = np.eye(4)
    link = model.GetLink(link_name)
    transform[:3, :3] = link.GetRotation()
    transform[:3, -1] = link.GetPosition()
    return transform


def load_joint_log(load_dir):
    csv_path = os.path.join(load_dir, "joint_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing joint log: {csv_path}")

    q_list = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            q_list.append(np.array([float(v) for v in row], dtype=float))

    if not q_list:
        raise ValueError(f"No joint rows found in {csv_path}")
    return q_list


def load_raw_scan_pcds(load_dir, excavator_model):
    q_list = load_joint_log(load_dir)
    scan_pcds = []

    for scan_idx, q in enumerate(q_list):
        q_state = np.concatenate([q, np.zeros(2)])
        excavator_model.SetState(q_state)

        scan_pcd = o3d.geometry.PointCloud()
        for lidar_idx, link_name in enumerate(LIDAR_LINK_NAMES, start=1):
            pcd_path = os.path.join(
                load_dir, f"field_scan_lidar{lidar_idx}_{scan_idx}_raw.pcd"
            )
            if not os.path.exists(pcd_path):
                raise FileNotFoundError(f"Missing raw lidar scan: {pcd_path}")

            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd.transform(get_link_transform(excavator_model, link_name))
            scan_pcd += pcd

        scan_pcds.append(scan_pcd)

    return scan_pcds


def merge_scan_pcds(scan_pcds):
    scene_pcd = o3d.geometry.PointCloud()
    for pcd in scan_pcds:
        scene_pcd += pcd
    return scene_pcd


def load_scene_pcd(args, excavator_model):
    if args.scan_source == "field_scan":
        print("Loading merged field scan from field_scan.pcd")
        return o3d.io.read_point_cloud(os.path.join(args.load_dir, "field_scan.pcd"))

    scan_pcds = load_raw_scan_pcds(args.load_dir, excavator_model)
    if args.scan_source == "raw_pose_graph":
        print(
            f"Refining scan alignment with pose graph optimization with {len(scan_pcds)} pcds"
        )
        transforms = refine_pose_graph(
            scan_pcds,
            voxel=args.pose_graph_voxel,
            max_correspondence_distance=args.pose_graph_max_correspondence,
            fitness_threshold=args.pose_graph_fitness_threshold,
            sequential_fitness_threshold=args.pose_graph_sequential_fitness_threshold,
            n_workers=args.pose_graph_workers,
            verbose=args.verbose,
        )
        scan_pcds_optimized = []
        for pcd, transform in zip(scan_pcds, transforms):
            pcd_optimized = copy.deepcopy(pcd)
            pcd_optimized.transform(transform)
            scan_pcds_optimized.append(pcd_optimized)
        scan_pcds = scan_pcds_optimized

    scene_pcd = merge_scan_pcds(scan_pcds)
    if args.write_scene_pcd:
        o3d.io.write_point_cloud(
            os.path.join(args.load_dir, args.write_scene_pcd), scene_pcd
        )
    return scene_pcd


def make_text_mesh(text, position, scale=0.05, color=(1, 0, 0)):
    text_mesh = o3d.t.geometry.TriangleMesh.create_text(str(text))
    text_mesh = text_mesh.scale(scale, center=[0, 0, 0])
    text_legacy = text_mesh.to_legacy()
    text_legacy.paint_uniform_color(color)
    text_legacy.rotate(TEXT_LABEL_ROTATION, center=[0, 0, 0])
    # TEXT_LABEL_ROTATION has det=-1, which reverses triangle winding and flips
    # face normals away from the camera. Swap each triangle's last two indices to
    # restore outward-facing normals for single-sided rendering.
    triangles = np.asarray(text_legacy.triangles).copy()
    triangles[:, [1, 2]] = triangles[:, [2, 1]]
    text_legacy.triangles = o3d.utility.Vector3iVector(triangles)
    text_legacy.compute_vertex_normals()
    return text_legacy.translate(position)


def build_default_cluster_ids(available_ids, preferred_ids, n_clusters):
    available_ids = [int(sid) for sid in available_ids]
    available_set = set(available_ids)
    defaults = []
    seen = set()

    for sid in preferred_ids:
        sid = int(sid)
        if sid in available_set and sid not in seen:
            defaults.append(sid)
            seen.add(sid)
        if len(defaults) >= n_clusters:
            return defaults

    for sid in available_ids:
        if sid not in seen:
            defaults.append(sid)
            seen.add(sid)
        if len(defaults) >= n_clusters:
            return defaults

    return defaults


def assign_cluster_ids_gui(
    cluster_pcds, available_ids, default_ids, preview_geometries=None
):
    """Returns (ids, remove_set) where remove_set is a set of cluster indices to drop."""
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QMessageBox,
            QPushButton,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )
    except Exception as exc:
        print(f"PySide6 GUI unavailable ({exc}); using default ids.")
        return default_ids[: len(cluster_pcds)], set()

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    available_ids = [int(sid) for sid in available_ids]
    default_ids = [int(sid) for sid in default_ids]

    # Precompute label anchor positions (2 m above each cluster centre).
    label_positions = []
    for cluster in cluster_pcds:
        pos = np.asarray(cluster.points).mean(0).copy()
        pos[2] += 2.0
        label_positions.append(pos)

    preview_vis = None
    preview_timer = None
    label_meshes = []  # one live label mesh per cluster
    preview_pcd_refs = list(
        cluster_pcds
    )  # tracks the geometry currently shown per cluster

    if preview_geometries is not None:
        preview_vis = o3d.visualization.Visualizer()
        preview_vis.create_window(
            window_name="Cluster Preview - assign IDs in the GUI",
            width=1280,
            height=720,
        )
        for geom in preview_geometries:
            preview_vis.add_geometry(geom)

        # Create initial labels showing only the cluster index (no default stone ID).
        for i, pos in enumerate(label_positions):
            mesh = make_text_mesh(str(i), pos)
            label_meshes.append(mesh)
            preview_vis.add_geometry(mesh, reset_bounding_box=False)

        preview_opt = preview_vis.get_render_option()
        preview_opt.background_color = np.asarray([0.9, 0.9, 0.9])
        preview_opt.point_size = 3.0

        def poll_preview():
            if preview_vis is None:
                return
            preview_vis.poll_events()
            preview_vis.update_renderer()

        preview_timer = QTimer()
        preview_timer.timeout.connect(poll_preview)
        preview_timer.start(30)

    dialog = QDialog()
    dialog.setWindowTitle("Assign Stone IDs to Clusters")
    dialog.resize(760, 520)

    layout = QVBoxLayout(dialog)
    layout.addWidget(
        QLabel(
            "Assign one stone id to each detected cluster. "
            "Rows follow the numbered cluster preview."
        )
    )

    table = QTableWidget(len(cluster_pcds), 6)
    table.setHorizontalHeaderLabels(
        ["Cluster", "Points", "Center X", "Center Y", "Stone ID", "Remove?"]
    )
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    layout.addWidget(table)

    combos = []
    remove_checks = []
    for row, cluster in enumerate(cluster_pcds):
        pts = np.asarray(cluster.points)
        center = pts.mean(0)
        values = [
            str(row),
            str(len(pts)),
            f"{center[0]:.3f}",
            f"{center[1]:.3f}",
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, col, item)

        combo = QComboBox()
        combo.addItem("-", None)
        for sid in available_ids:
            combo.addItem(str(sid), sid)
        table.setCellWidget(row, 4, combo)
        combos.append(combo)

        chk = QCheckBox()
        chk.setStyleSheet("margin-left: auto; margin-right: auto;")
        table.setCellWidget(row, 5, chk)
        remove_checks.append(chk)

    # Wire each combo so changing the stone ID refreshes the 3-D label live.
    def make_label_updater(row):
        def _update(_index=None):
            if preview_vis is None or row >= len(label_meshes):
                return
            stone_id = combos[row].currentData()
            label_text = f"{row}:{stone_id}" if stone_id is not None else str(row)
            preview_vis.remove_geometry(label_meshes[row], reset_bounding_box=False)
            new_mesh = make_text_mesh(label_text, label_positions[row])
            label_meshes[row] = new_mesh
            preview_vis.add_geometry(new_mesh, reset_bounding_box=False)

        return _update

    for row, combo in enumerate(combos):
        combo.currentIndexChanged.connect(make_label_updater(row))

    # Wire each checkbox so toggling it hides/shows the cluster and its label in the preview.
    def make_remove_toggler(row):
        def _toggle(state):
            if preview_vis is None:
                return
            removing = bool(state)
            if removing:
                preview_vis.remove_geometry(
                    preview_pcd_refs[row], reset_bounding_box=False
                )
                if row < len(label_meshes):
                    preview_vis.remove_geometry(
                        label_meshes[row], reset_bounding_box=False
                    )
            else:
                preview_vis.add_geometry(
                    preview_pcd_refs[row], reset_bounding_box=False
                )
                if row < len(label_meshes):
                    preview_vis.add_geometry(
                        label_meshes[row], reset_bounding_box=False
                    )

        return _toggle

    for row, chk in enumerate(remove_checks):
        chk.stateChanged.connect(make_remove_toggler(row))

    # "Remove Points" button — opens a PyVista picker for the selected cluster row.
    remove_pts_btn = QPushButton("Remove Points from Selected Cluster")

    def on_remove_points_clicked():
        row = table.currentRow()
        if row < 0:
            QMessageBox.information(
                dialog, "No selection", "Select a cluster row first."
            )
            return

        if preview_timer is not None:
            preview_timer.stop()

        old_pcd = cluster_pcds[row]
        new_pcd = manually_remove_points(old_pcd, distance=0.05, max_remove_ratio=0.8)

        if new_pcd is not old_pcd and len(np.asarray(new_pcd.points)) > 0:
            # Preserve the cluster colour.
            old_colors = np.asarray(old_pcd.colors)
            if len(old_colors) > 0:
                new_pcd.paint_uniform_color(old_colors[0])

            # Swap geometry in the Open3D preview.
            if preview_vis is not None:
                preview_vis.remove_geometry(
                    preview_pcd_refs[row], reset_bounding_box=False
                )
                preview_vis.add_geometry(new_pcd, reset_bounding_box=False)

            cluster_pcds[row] = new_pcd
            preview_pcd_refs[row] = new_pcd

            # Recompute label anchor and refresh the 3-D label.
            new_pos = np.asarray(new_pcd.points).mean(0).copy()
            new_pos[2] += 2.0
            label_positions[row] = new_pos
            make_label_updater(row)()

            # Update the point-count cell in the table.
            pts_item = QTableWidgetItem(str(len(np.asarray(new_pcd.points))))
            pts_item.setFlags(pts_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 1, pts_item)

        if preview_timer is not None:
            preview_timer.start(30)

    remove_pts_btn.clicked.connect(on_remove_points_clicked)

    edit_row = QHBoxLayout()
    edit_row.addWidget(remove_pts_btn)
    layout.addLayout(edit_row)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(buttons)

    def accept_if_valid():
        kept = [i for i, chk in enumerate(remove_checks) if not chk.isChecked()]
        selected = [combos[i].currentData() for i in kept if combos[i].currentData() is not None]
        duplicates = sorted({sid for sid in selected if selected.count(sid) > 1})
        if duplicates:
            QMessageBox.warning(
                dialog,
                "Duplicate Stone IDs",
                "Each kept cluster must have a unique stone id. "
                f"Duplicates: {duplicates}",
            )
            return
        dialog.accept()

    buttons.accepted.connect(accept_if_valid)
    buttons.rejected.connect(dialog.reject)

    try:
        if dialog.exec() != QDialog.Accepted:
            raise RuntimeError("Stone id assignment cancelled.")
        remove_set = {i for i, chk in enumerate(remove_checks) if chk.isChecked()}
        # Clusters left at "-" (no stone assigned) are treated as removed.
        for i, combo in enumerate(combos):
            if combo.currentData() is None:
                remove_set.add(i)
        return [combo.currentData() for combo in combos], remove_set
    finally:
        if preview_timer is not None:
            preview_timer.stop()
        if preview_vis is not None:
            preview_vis.destroy_window()


if __name__ == "__main__":
    rclpy.init()
    pose_array_pub = PoseArrayPublisher()

    origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default=".data/field_pcd/260519_9")
    parser.add_argument(
        "--scan_source",
        choices=["field_scan", "raw", "raw_pose_graph"],
        default="raw_pose_graph",
        help=(
            "field_scan: load field_scan.pcd; raw: transform and merge raw lidar "
            "PCDs using joint_log.csv; raw_pose_graph: additionally refine scan "
            "alignment with pose graph optimization."
        ),
    )
    parser.add_argument("--write_scene_pcd", default="")
    parser.add_argument("--pose_graph_voxel", type=float, default=0.03)
    parser.add_argument("--pose_graph_max_correspondence", type=float, default=0.06)
    parser.add_argument("--pose_graph_fitness_threshold", type=float, default=0.35)
    parser.add_argument(
        "--pose_graph_sequential_fitness_threshold", type=float, default=0.15
    )
    parser.add_argument("--pose_graph_workers", type=int, default=4)
    parser.add_argument(
        "--icp_min_fitness",
        type=float,
        default=0.05,
        help=(
            "Minimum final ICP fitness required to keep a fitted stone. "
            "Lower-fitness meshes are treated as unaligned to the cluster and skipped."
        ),
    )
    parser.add_argument("--add_mesh_points", action="store_true")
    parser.add_argument(
        "--no_id_gui",
        action="store_true",
        help="Use the hard-coded/default stone-id ordering instead of the assignment GUI.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=None,
        help="Ordered list of stone IDs to assign to clusters (overrides preferred_ids).",
    )
    parser.add_argument(
        "--remove_clusters",
        nargs="*",
        type=int,
        default=None,
        help="Cluster indices to remove before ICP.",
    )
    parser.add_argument(
        "--manual_remove",
        action="store_true",
        help="Open a PyVista point picker to manually remove noise from the scene PCD before clustering.",
    )
    parser.add_argument(
        "--manual_remove_distance",
        type=float,
        default=0.05,
        help="Removal radius around each picked point (default 0.05 m).",
    )
    parser.add_argument(
        "--manual_init_poses",
        nargs="+",
        default=[],
        metavar="STONE_ID:x,y,z,qx,qy,qz,qw",
        help=(
            "Pin the ICP initial pose for specific stone IDs. "
            "Format: STONE_ID:x,y,z,qx,qy,qz,qw (world-frame pose, quat xyzw). "
            "Repeat to set multiple stones, e.g. --manual_init_poses 3:1.2,0.5,0.1,0,0,0,1 9:..."
        ),
    )
    parser.add_argument(
        "--init_pose_data",
        nargs="?",
        const="",
        default=None,
        metavar="PKL",
        help=(
            "Load ICP initial poses from pose_data.pkl. If PKL is omitted, "
            "use <load_dir>/pose_data.pkl. Explicit --manual_init_poses "
            "override loaded pose-data entries."
        ),
    )
    parser.add_argument(
        "--manual_init_gui",
        action="store_true",
        help="Open an interactive 3-D viewer to adjust initial poses before ICP.",
    )
    parser.add_argument(
        "--manual_init_gui_stones",
        nargs="+",
        type=int,
        default=None,
        metavar="STONE_ID",
        help=(
            "Restrict the manual init GUI to these stone IDs. "
            "e.g. --manual_init_gui_stones 3 9 14. "
            "If omitted, all assigned stones are shown."
        ),
    )
    parser.add_argument(
        "--manual_init_translation_step",
        type=float,
        default=0.05,
        help="Translation step size in metres for the manual init GUI (default: 0.05).",
    )
    parser.add_argument(
        "--manual_init_rotation_step",
        type=float,
        default=5.0,
        help="Rotation step size in degrees for the manual init GUI (default: 5.0).",
    )
    args = parser.parse_args()
    load_dir = args.load_dir

    manual_init_poses: dict[int, np.ndarray] = {}
    if args.init_pose_data is not None:
        pose_data_path = (
            os.path.join(load_dir, "pose_data.pkl")
            if args.init_pose_data == ""
            else args.init_pose_data
        )
        pose_data_poses = load_pose_data_initial_poses(pose_data_path)
        manual_init_poses.update(pose_data_poses)
        print(
            f"Loaded {len(pose_data_poses)} initial poses from pose data: "
            f"{pose_data_path}"
        )
    for entry in args.manual_init_poses:
        stone_id, pose = parse_stone_id_pose(entry)
        manual_init_poses[stone_id] = pose
        print(f"Manual init pose for stone {stone_id}: {pose.tolist()}")

    cfg = omegaconf.OmegaConf.load("agent/configs/config.yml")
    excavator_model, _ = get_excavator_model()
    stone_meshes, _, pcds, _ = get_stone_model()
    available_stone_ids = sorted(pcds.keys())
    pcds_vis = copy.deepcopy(pcds)

    scene_pcd = load_scene_pcd(args, excavator_model)
    scene_pcd = scene_pcd.voxel_down_sample(0.01)
    scene_pcd = box_crop_largest_cluster(
        # scene_pcd, [-10, 10], [-10, 0], [-1, 2], cluster=False
        scene_pcd,
        [-3.0, 8],
        [-8, 9],
        [0, 2],
        cluster=False,
    )
    min_bound = np.array([-3.0, -2, -1.0])
    max_bound = np.array([2.5, 2.0, 5.0])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    scene_pcd = remove_points_inside_aabb(scene_pcd, bbox)

    if args.manual_remove:
        scene_pcd = manually_remove_points(
            scene_pcd,
            distance=args.manual_remove_distance,
            max_remove_ratio=0.9,
        )

    o3d.visualization.draw_geometries([scene_pcd, origin_coord])

    plane_model, inliers = scene_pcd.segment_plane(
        distance_threshold=GLOBAL_RANSAC_THRESHOLD, ransac_n=10, num_iterations=1000
    )
    stone_pcd = scene_pcd.select_by_index(inliers, invert=True)
    stone_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    labels = np.array(
        stone_pcd.cluster_dbscan(
            eps=GLOBAL_CLUSTER_THRESHOLD, min_points=50, print_progress=False
        )
    )

    max_label = labels.max()
    print(f"Cluster count: {max_label + 1}")

    cluster_pcds = []

    for cluster_id in range(max_label + 1):
        idx = np.where(labels == cluster_id)[0]
        cluster = stone_pcd.select_by_index(idx)
        if len(cluster.points) > 500:
            cluster_pcds.append(cluster)

    cluster_pcds = sorted(cluster_pcds, key=lambda c: len(c.points), reverse=True)
    cluster_pcds = cluster_pcds[: len(available_stone_ids)]

    # Local RANSAC refinement: the global plane fit needs a large threshold
    # (~0.10 m) because the ground is not flat, but that eats the lower part of
    # each stone. Re-fit the plane locally around each cluster with a smaller
    # threshold so the recovered stone pcd keeps its base.
    neighborhood_margin_xy = 0.20
    downward_margin = 0.30
    refined_cluster_pcds = []
    for ci, cluster in enumerate(cluster_pcds):
        cluster_pts = np.asarray(cluster.points)
        original_center = cluster_pts.mean(0)
        n_global = len(cluster_pts)

        min_b = cluster_pts.min(0) - np.array(
            [neighborhood_margin_xy, neighborhood_margin_xy, 0.0]
        )
        max_b = cluster_pts.max(0) + np.array(
            [neighborhood_margin_xy, neighborhood_margin_xy, 0.0]
        )
        min_b[2] = cluster_pts[:, 2].min() - downward_margin

        local_bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
        local_pcd = scene_pcd.crop(local_bbox)

        if len(local_pcd.points) < 100:
            print(
                f"[refine] cluster {ci}: local crop too small ({len(local_pcd.points)} pts) → kept as-is"
            )
            refined_cluster_pcds.append(cluster)
            continue

        plane_model, local_inliers = local_pcd.segment_plane(
            distance_threshold=LOCAL_RANSAC_THRESHOLD,
            ransac_n=5,
            num_iterations=500,
        )
        # Reject the fitted plane if it is not roughly horizontal (ground planes
        # have normals close to vertical; a tilted plane hitting a stone's flat
        # top face will have a near-horizontal normal and should be ignored).
        plane_normal = np.abs(plane_model[:3])
        plane_normal /= np.linalg.norm(plane_normal)
        is_ground_plane = plane_normal[2] >= 0.9  # normal Z-component ≥ cos(45°)
        if is_ground_plane:
            local_stone_pcd = local_pcd.select_by_index(local_inliers, invert=True)
        else:
            print(
                f"[refine] cluster {ci}: local plane normal {plane_normal.round(3).tolist()} "
                f"is not horizontal — skipping local RANSAC removal"
            )
            local_stone_pcd = local_pcd
        n_after_ransac = len(np.asarray(local_stone_pcd.points))
        print(
            f"[refine] cluster {ci} @ {original_center.round(2).tolist()}: "
            f"global={n_global}  local_crop={len(local_pcd.points)}  "
            f"after_local_ransac={n_after_ransac}  "
            f"removed_by_ransac={len(local_pcd.points)-n_after_ransac}"
        )

        if len(local_stone_pcd.points) < 100:
            print(f"[refine] cluster {ci}: after local RANSAC too small → kept as-is")
            refined_cluster_pcds.append(cluster)
            continue

        local_labels = np.array(
            local_stone_pcd.cluster_dbscan(
                eps=LOCAL_CLUSTER_THRESHOLD, min_points=20, print_progress=False
            )
        )
        if local_labels.max() < 0:
            refined_cluster_pcds.append(local_stone_pcd)
            continue

        # Pick the sub-cluster whose centroid is closest to the original
        # cluster's centroid so a neighboring stone caught by the bbox does not
        # hijack the result.
        local_pts = np.asarray(local_stone_pcd.points)
        best_label, best_dist = -1, float("inf")
        for lbl in range(local_labels.max() + 1):
            idx = np.where(local_labels == lbl)[0]
            if len(idx) < 50:
                continue
            d = np.linalg.norm(local_pts[idx].mean(0) - original_center)
            if d < best_dist:
                best_dist = d
                best_label = lbl

        if best_label < 0:
            print(f"[refine] cluster {ci}: no valid sub-cluster found → kept as-is")
            refined_cluster_pcds.append(cluster)
        else:
            idx = np.where(local_labels == best_label)[0]
            refined = local_stone_pcd.select_by_index(idx)
            n_final = len(np.asarray(refined.points))
            print(
                f"[refine] cluster {ci}: sub-cluster selected "
                f"(label={best_label}, dist={best_dist:.3f}): "
                f"{n_after_ransac} → {n_final} pts  "
                f"removed_by_dbscan={n_after_ransac - n_final}"
            )
            refined.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            refined_cluster_pcds.append(refined)

    cluster_pcds = refined_cluster_pcds

    colors = np.random.rand(max_label + 1, 3)
    colored_points = [colors[label] if label >= 0 else [0, 0, 0] for label in labels]
    stone_pcd.colors = o3d.utility.Vector3dVector(colored_points)

    o3d.visualization.draw_geometries([stone_pcd])

    angles_x = [0, 90, 180, 270]
    angles_y = [0, 90, 180, 270]
    angles_z = [0, 90, 180, 270]

    angles_x, angles_y, angles_z = np.meshgrid(
        angles_x, angles_y, angles_z, indexing="ij"
    )
    angles_x = angles_x.flatten()
    angles_y = angles_y.flatten()
    angles_z = angles_z.flatten()
    rots = []
    for angle_x, angle_y, angle_z in zip(angles_x, angles_y, angles_z):
        rots.append(
            Rotation.from_euler(
                "xyz", [angle_x, angle_y, angle_z], degrees=True
            ).as_matrix()
        )

    poses = []

    def radial_sort_key(c):
        cx, cy = np.asarray(c.points).mean(0)[:2]
        radius = np.sqrt(cx**2 + cy**2)
        # Convention: +X=0, +Y=-π/2, -Y=+π/2 (counter-clockwise from +X)
        # atan2(-y, x) achieves this mapping.
        angle = np.arctan2(-cy, cx)
        return (radius, angle)

    cluster_pcds = sorted(cluster_pcds, key=radial_sort_key)

    cluster_pcds_recolored = []
    for i, lidar_pcd in enumerate(cluster_pcds):
        lidar_pcd.paint_uniform_color(np.array([i / len(cluster_pcds), 0.5, 0.5]))
        cluster_pcds_recolored.append(lidar_pcd)

    # ids = [13, 11, 19, 20, 9,  2,  6, 3, 8,  5, 4,  1, 15, 17, 18, 14] # 0519_2
    # ids = [9, 4, 11, 19, 6, 13, 14, 8, 5, 18, 1, 15, 20, 2, 17, 3]  # 0519_3
    # preferred_ids = [9, 11, 13, 19, 3, 6, 8, 5, 4, 2, 1, 15, 14, 20, 17, 18]  # 0521_4
    # preferred_ids = [9, 11, 3, 13, 8, 14, 19, 18, 5, 2, 1, 6, 4, 20, 17, 15] # 0522_2
    preferred_ids = (
        args.ids
        if args.ids is not None
        else [9, 11, 13, 3, 8, 14, 19, 5, 2, 18, 1, 4, 6, 20, 17, 15]
    )  # 0522_3
    ids = build_default_cluster_ids(
        available_stone_ids, preferred_ids, len(cluster_pcds)
    )

    cli_remove_set = set(args.remove_clusters) if args.remove_clusters else set()

    if not args.no_id_gui:
        # Labels are owned and updated live by assign_cluster_ids_gui; pass only
        # the static geometries here.
        ids, remove_set = assign_cluster_ids_gui(
            cluster_pcds,
            available_ids=available_stone_ids,
            default_ids=ids,
            preview_geometries=cluster_pcds + [origin_coord],
        )
        remove_set |= cli_remove_set
    else:
        cluster_labels = []
        for cluster_idx, cluster_pcd in enumerate(cluster_pcds):
            position = np.asarray(cluster_pcd.points).mean(0).copy()
            position[2] += 2.0
            label = (
                f"{cluster_idx}:{ids[cluster_idx]}"
                if cluster_idx < len(ids)
                else str(cluster_idx)
            )
            cluster_labels.append((position, label))
        _draw_with_labels(
            cluster_pcds + [origin_coord],
            cluster_labels,
            window_name="Cluster Preview",
        )
        if args.remove_clusters is not None and len(args.remove_clusters) == 0:
            raw = input(
                "Enter cluster indices to remove (space-separated, or Enter to skip): "
            ).strip()
            remove_set = {int(x) for x in raw.split() if x.lstrip("-").isdigit()}
        else:
            remove_set = cli_remove_set

    if remove_set:
        print(f"Removing clusters: {sorted(remove_set)}")
        cluster_pcds = [c for i, c in enumerate(cluster_pcds) if i not in remove_set]
        ids = [id for i, id in enumerate(ids) if i not in remove_set]

    if len(ids) < len(cluster_pcds):
        raise ValueError(
            f"Only {len(ids)} stone ids were assigned for "
            f"{len(cluster_pcds)} clusters."
        )

    id_labels = []
    for id, cluster_pcd in zip(ids, cluster_pcds):
        position = np.asarray(cluster_pcd.points).mean(0).copy()
        position[2] += 2.0
        id_labels.append((position, str(id)))
    _draw_with_labels(
        cluster_pcds + [origin_coord],
        id_labels,
        window_name="Assigned stone IDs",
    )

    if args.manual_init_gui:
        gui_poses = run_manual_init_gui(
            cluster_pcds,
            ids,
            stone_meshes,
            manual_init_poses,
            translation_step=args.manual_init_translation_step,
            rotation_step_deg=args.manual_init_rotation_step,
            target_stone_ids=(
                set(args.manual_init_gui_stones)
                if args.manual_init_gui_stones is not None
                else None
            ),
        )
        manual_init_poses.update(gui_poses)

    for i, lidar_pcd in enumerate(cluster_pcds):
        center = np.asarray(lidar_pcd.points).mean(0)
        print(f"clustered pcd {i}'s center {center.tolist()}")

    for i, (lidar_pcd, assigned_id) in enumerate(zip(cluster_pcds, ids)):
        print("Processing cluster {} with assigned id {}".format(i, assigned_id))

        icp_results = []
        center = np.asarray(lidar_pcd.points).mean(0)
        init_T = np.eye(4)
        init_T[:3, -1] = center
        if len(pcds) == 0:
            continue
        if assigned_id not in pcds:
            raise KeyError(
                f"Assigned stone id {assigned_id} is unavailable. "
                f"Remaining ids: {sorted(pcds.keys())}"
            )
        id = assigned_id
        pcd = copy.deepcopy(pcds[id])

        if args.add_mesh_points:
            pcd_mesh = stone_meshes[id].sample_points_uniformly(number_of_points=20000)
            pcd_mesh = remove_points_from_points(pcd, pcd_mesh, 0.05)
            pcd += pcd_mesh

        if id in manual_init_poses:
            manual_T = pose_to_matrix(manual_init_poses[id])
            iter_rots = [manual_T[:3, :3]]
            init_T[:3, -1] = manual_T[:3, 3]
            print(
                f"  stone {id}: using manual init pose {manual_init_poses[id].tolist()}"
            )
        else:
            iter_rots = rots

        for rot in iter_rots:
            init_T[:3, :3] = rot
            if POSEID_MODE == 0:
                target_T, history = multiscale_icp(
                    lidar_pcd,
                    pcd,
                    np.linalg.inv(init_T),
                    voxel_sizes=[0.1, 0.05, 0.02, 0.01],
                    max_iters=[50, 30, 14, 7],
                )
                target_T = np.linalg.inv(target_T)
            else:
                target_T, history = multiscale_icp(
                    pcd,
                    lidar_pcd,
                    init_T,
                    voxel_sizes=[0.1, 0.05, 0.02, 0.01],
                    max_iters=[50, 30, 14, 7],
                )

            mesh = copy.deepcopy(stone_meshes[id])
            mesh.transform(target_T)
            v = np.asarray(mesh.vertices)
            z_min = v[:, 2].min()
            if z_min > Z_THRESHOLD:
                icp_results.append(
                    (id, target_T, z_min, history[-1][1] - 0.0 * history[-1][2])
                )
            elif args.verbose:
                print(
                    f"ICP for {id} with center {center.tolist()} and rot \n"
                    f"{rot} \n is rejected due to low z_min {z_min}"
                )

        if not icp_results:
            print(
                f"No valid ICP result for cluster {i} with assigned id {id}; skipping."
            )
            continue

        best_result = max(icp_results, key=lambda x: x[-1])
        id, target_T, z_min, fitness = best_result
        if fitness < args.icp_min_fitness:
            print(
                f"ICP for {id} rejected: fitness {fitness:.3f} "
                f"< {args.icp_min_fitness:.3f}; mesh is not aligned to cluster {i}."
            )
            continue
        stone_meshes[id].transform(target_T)
        pcds_vis[id].transform(target_T)
        poses.append(
            (target_T[:3, -1], Rotation.from_matrix(target_T[:3, :3]).as_quat(), id)
        )
        pcds.pop(id)
        if args.verbose:
            print(
                f"ICP for {id} with center {center.tolist()}, fitness {fitness} and z_min {z_min} is done"
            )
    if args.verbose:
        for pos, quat, id in poses:
            print(f"model_{id} - pos = {pos.tolist()}, quat = {quat.tolist()}")

    fitted_stone_ids = [stone_id for _, _, stone_id in poses]
    plane_width, plane_length = 30.0, 30.0
    plane = o3d.geometry.TriangleMesh.create_box(plane_width, plane_length, 0.001)
    plane = plane.translate([-plane_width / 2, -plane_length / 2, -0.001])
    fitted_labels = []
    for stone_id in fitted_stone_ids:
        bbox = stone_meshes[stone_id].get_axis_aligned_bounding_box()
        pos = bbox.get_center().copy()
        pos[2] = bbox.get_max_bound()[2] + 0.4
        fitted_labels.append((pos, str(stone_id)))

    _draw_with_labels(
        [stone_meshes[stone_id] for stone_id in fitted_stone_ids]
        + [origin_coord, plane]
        + cluster_pcds,
        fitted_labels,
        window_name="Fitted stone meshes",
    )
    _draw_with_labels(
        [pcds_vis[stone_id] for stone_id in fitted_stone_ids]
        + [origin_coord, plane]
        + cluster_pcds,
        fitted_labels,
        window_name="Fitted stone PCDs",
    )

    poses_dict = {stone_id: (pos, quat) for pos, quat, stone_id in poses}
    with open(os.path.join(load_dir, "pose_data.pkl"), "wb") as f:
        pickle.dump(poses_dict, f)

    for _ in range(100):
        pose_array_pub.publish(poses)
