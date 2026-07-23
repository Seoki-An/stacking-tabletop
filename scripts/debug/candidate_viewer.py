#!/usr/bin/env python3
"""
Interactive MCTS candidate viewer.

Usage:
    python scripts/debug/candidate_viewer.py <debug_state.pkl>
    python scripts/debug/candidate_viewer.py .debug/mcts/<run>/debug_state.pkl
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue
import sys
import threading
from pathlib import Path

import imageio
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import pose_to_transformation_matrix
from utils.dsf import DiffSupportSimple
from utils.geometry import fix_normals_outward
from utils.wavefront import WavefrontImporter
from agent.env import StoneStackingEnv
from agent.env.components.state import State
from scripts.debug.mcts_map_images import compact_score_map_debug, state_score_map_debug

_PANEL_W = 350

_COLOR_GROUND        = [0.48, 0.58, 0.42, 1.00]
_COLOR_SCENEID_GROUND = [0.05, 0.55, 0.95, 0.70]
_COLOR_SCENEID_PLACE_BOX = [0.05, 0.55, 0.95, 0.18]
_COLOR_ACTIVE_FLOOR_OCCUPIED = [0.10, 0.75, 0.95, 0.32]
_COLOR_ACTIVE_FLOOR_GRID = [0.10, 0.75, 0.95, 0.90]
_COLOR_SCORE_SELECTED_XY = [1.00, 1.00, 1.00, 1.00]
_COLOR_SCORE_VALID_CELL = [0.88, 0.88, 0.88, 0.16]
_COLOR_WALL          = [0.78, 0.78, 0.88, 0.12]
_COLOR_SCENE_STONE   = [0.50, 0.50, 0.50, 1.00]
_COLOR_CAND_SELECTED = [1.00, 0.80, 0.05, 1.00]  # amber — currently selected
_COLOR_CAND_INITIAL  = [0.92, 0.92, 0.92, 1.00]
_COLOR_CAND_SOLVED   = [0.10, 0.85, 0.25, 1.00]
_COLOR_CAND_FAILED   = [0.25, 0.25, 0.25, 0.35]  # dark gray — failed candidates
_COLOR_TRAJECTORY    = [1.00, 0.55, 0.05, 0.22]
_COLOR_TRAJ_LINE     = [1.00, 0.25, 0.05, 1.00]
_COLOR_TRAJ_START    = [0.10, 0.80, 0.25, 1.00]
_COLOR_TRAJ_END      = [1.00, 0.10, 0.10, 1.00]
_COLOR_SCENE_MOTION  = [0.95, 0.10, 0.18, 0.78]
_COLOR_SCENE_MOTION_LINE = [1.00, 0.05, 0.35, 1.00]
_COLOR_CONTACT       = [0.10, 0.95, 1.00, 1.00]
_COLOR_CONTACT_NORMAL = [0.10, 0.95, 1.00, 1.00]
_COLOR_POSE_SOLVE_CONTACT = [0.95, 0.20, 0.95, 1.00]
_COLOR_POSE_SOLVE_FORCE = [1.00, 0.08, 0.80, 1.00]
_COLOR_FAILED_GRASP  = [0.90, 0.15, 0.15, 0.30]  # transparent red — failed grasps
_COLOR_SEQUENCE_FUTURE = [0.65, 0.30, 1.00, 0.42]
_COLOR_EXCAVATOR     = [0.95, 0.82, 0.28, 1.00]
_GROUND_MARGIN = 4.0
_GROUND_DEPTH = 0.001
_PLACE_HEIGHT_BOX_MARGIN = 0.25
_PLACE_HEIGHT_BOX_MIN_THICKNESS = 0.04
_SCORE_MAP_Z_CLEARANCE = 0.20
_DEFAULT_TRAJECTORY_VIDEO_FPS = 30.0
_TRAJECTORY_VIDEO_SUBDIR = Path("videos") / "candidate_trajectories"
_LIDAR_FRAME_LINKS = ("lidar1_link", "lidar2_link", "lidar3_link")
_END_EFFECTOR_LINK = "grip_body"

_EXCAVATOR_JOINTS = [
    ("Swing", 0, -180.0, 180.0),
    ("Boom", 1, -30.0, 120.0),
    ("Arm", 2, -180.0, 60.0),
    ("Bucket", 3, -180.0, 180.0),
    ("Tilt", 4, -180.0, 180.0),
    ("Rotate", 5, -360.0, 360.0),
    ("Grip L", 6, -90.0, 90.0),
    ("Grip R", 7, -90.0, 90.0),
]


def _ablation_result_path_for_state(state_path: Path) -> Path:
    state_path = Path(state_path)
    for parent in state_path.parents:
        if parent.name != "states":
            continue
        relative = state_path.relative_to(parent)
        if len(relative.parts) < 2:
            break
        result_path = parent.parent / f"{relative.parts[0]}.json"
        if result_path.is_file():
            return result_path
        break
    raise ValueError(
        "bare State pickle is not inside an ablation case with a matching "
        "<config>.json result"
    )


def _geometry_arrays(geometries) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles, offset = [], [], 0
    for geometry in geometries:
        mesh = geometry.get_mesh()
        mesh_vertices = np.asarray(mesh.vertices, dtype=float)
        mesh_triangles = np.asarray(mesh.triangles, dtype=int)
        vertices.append(mesh_vertices)
        triangles.append(mesh_triangles + offset)
        offset += len(mesh_vertices)
    if not vertices:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)
    return np.concatenate(vertices), np.concatenate(triangles)


def _debug_data_from_ablation_state(state_path: Path, state: State) -> dict:
    result_path = _ablation_result_path_for_state(state_path)
    result = json.loads(result_path.read_text())
    environment = result.get("settings", {}).get("environment_config")
    if not isinstance(environment, dict):
        raise ValueError(f"ablation result has no environment snapshot: {result_path}")

    env_cfg = OmegaConf.create(environment)
    env_cfg.data.fixed_stone_set = [
        int(stone_id) for stone_id in np.asarray(state.stone_set).reshape(-1)
    ]
    env_cfg.n_stone = len(env_cfg.data.fixed_stone_set)
    env = StoneStackingEnv(
        {"cfg": env_cfg, "n_threads": 1, "build_action_builder": False}
    )
    env.reset()
    env.update_from_state(state)

    stone_meshes = {
        int(stone.id): _geometry_arrays(stone.geometries)
        for stone in env.inventory.stones
    }
    wall = env.inventory.target_wall
    wall_meshes = [
        _geometry_arrays([geometry]) for geometry in wall.geometries
    ]
    stone_set = [int(v) for v in np.asarray(state.stone_set).reshape(-1)]
    stone_seq = [stone_set[int(index)] for index in state.stone_seq]
    stone_poses = {
        int(stone_id): np.asarray(pose, dtype=float).copy()
        for stone_id, pose in state.stone_poses.items()
    }
    load_dir = str(env_cfg.data.get("load_dir", ""))
    return {
        "target_wall_cfg": {
            "width": float(wall.width),
            "length": float(wall.length),
            "height": float(wall.height),
        },
        "target_wall_meshes": wall_meshes,
        "stone_meshes": stone_meshes,
        "stone_ply_meshes": {},
        "mesh_source": "dsf",
        "asset_dir": load_dir,
        "steps": [
            {
                "step": len(state.stone_seq),
                "succeeded": not bool(state.failed),
                "state_only": True,
                "scene": {
                    "stone_seq": stone_seq,
                    "stone_poses": stone_poses,
                },
                "score_map": state_score_map_debug(env, state),
                "candidates": [],
                "raw_state": state,
                "resume_state": state,
            }
        ],
        "resume_state": state,
        "resume_step": len(state.stone_seq),
    }


def _load_viewer_data(data_path: str | Path) -> dict:
    path = Path(data_path)
    with path.open("rb") as file:
        loaded = pickle.load(file)
    if isinstance(loaded, dict):
        return loaded
    if isinstance(loaded, State):
        return _debug_data_from_ablation_state(path, loaded)
    raise TypeError(
        f"unsupported candidate-viewer pickle type: {type(loaded).__name__}"
    )


def _score_color(score: float, lo: float, hi: float) -> np.ndarray:
    """Blue→cyan→yellow→red gradient matching the mcts.py candidate PNG palette."""
    if not np.isfinite(score):
        return np.array(_COLOR_CAND_FAILED)
    t = 0.5 if hi <= lo else (score - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    anchors = np.array([
        [0.20, 0.32, 0.85, 0.42],
        [0.00, 0.78, 0.82, 0.48],
        [0.98, 0.86, 0.20, 0.55],
        [0.90, 0.18, 0.16, 0.65],
    ])
    x = t * (len(anchors) - 1)
    i = min(int(np.floor(x)), len(anchors) - 2)
    return anchors[i] * (i + 1 - x) + anchors[i + 1] * (x - i)


def _make_mesh(verts: np.ndarray, tris: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
    mesh.compute_vertex_normals()
    return mesh


def _mesh_renderable(mesh: o3d.geometry.TriangleMesh) -> bool:
    if mesh.is_empty():
        return False
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)
    if vertices.size == 0 or triangles.size == 0:
        return False
    if not np.all(np.isfinite(vertices)):
        return False
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    return bool(np.all(np.isfinite(extent)) and np.any(extent > 1e-9))


def _geometry_renderable(geom) -> bool:
    if isinstance(geom, o3d.geometry.TriangleMesh):
        return _mesh_renderable(geom)
    if isinstance(geom, o3d.geometry.LineSet):
        points = np.asarray(geom.points, dtype=float)
        lines = np.asarray(geom.lines, dtype=int)
        if points.size == 0 or lines.size == 0:
            return False
        if not np.all(np.isfinite(points)):
            return False
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        return bool(np.all(np.isfinite(extent)) and np.any(extent > 1e-9))
    return False


def _mat(color: list, transparent: bool = False) -> rendering.MaterialRecord:
    m = rendering.MaterialRecord()
    m.base_color = color
    if transparent or color[3] < 1.0:
        m.shader = "defaultLitTransparency"
        m.has_alpha = True
    else:
        m.shader = "defaultLit"
    return m


def _line_mat(color: list, width: float = 4.0) -> rendering.MaterialRecord:
    m = rendering.MaterialRecord()
    m.shader = "unlitLine"
    m.base_color = color
    m.line_width = width
    return m


def _video_mesh_entry(
    name: str,
    mesh: o3d.geometry.TriangleMesh,
    color: list,
    transparent: bool = False,
) -> dict | None:
    if not _mesh_renderable(mesh):
        return None
    return {
        "kind": "mesh",
        "name": str(name),
        "vertices": np.asarray(mesh.vertices, dtype=np.float64).copy(),
        "triangles": np.asarray(mesh.triangles, dtype=np.int32).copy(),
        "color": list(color),
        "transparent": bool(transparent),
    }


def _video_line_entry(
    name: str,
    line_set: o3d.geometry.LineSet,
    color: list,
    width: float = 4.0,
) -> dict | None:
    if not _geometry_renderable(line_set):
        return None
    return {
        "kind": "line",
        "name": str(name),
        "points": np.asarray(line_set.points, dtype=np.float64).copy(),
        "lines": np.asarray(line_set.lines, dtype=np.int32).copy(),
        "color": list(color),
        "width": float(width),
    }


def _add_video_entry(scene, entry: dict) -> None:
    kind = entry.get("kind")
    name = str(entry.get("name", "entry"))
    if kind == "line":
        geom = o3d.geometry.LineSet()
        geom.points = o3d.utility.Vector3dVector(entry["points"])
        geom.lines = o3d.utility.Vector2iVector(entry["lines"])
        mat = _line_mat(
            entry.get("color", [1.0, 1.0, 1.0, 1.0]),
            entry.get("width", 4.0),
        )
    else:
        geom = o3d.geometry.TriangleMesh()
        geom.vertices = o3d.utility.Vector3dVector(entry["vertices"])
        geom.triangles = o3d.utility.Vector3iVector(entry["triangles"])
        geom.compute_vertex_normals()
        color = entry.get("color", [0.7, 0.7, 0.7, 1.0])
        mat = _mat(color, transparent=bool(entry.get("transparent", False)))
    if _geometry_renderable(geom):
        scene.add_geometry(name, geom, mat)


def _candidate_trajectory_video_process(job: dict, result_queue) -> None:
    try:
        width = int(job["width"])
        height = int(job["height"])
        renderer = rendering.OffscreenRenderer(width, height)
        scene = renderer.scene
        try:
            scene.set_background([0.12, 0.12, 0.12, 1.0])
            scene.scene.set_indirect_light_intensity(30000)
        except Exception:
            pass

        for entry in job.get("static_entries", []):
            _add_video_entry(scene, entry)

        target = job["target_mesh"]
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = o3d.utility.Vector3dVector(target["vertices"])
        target_mesh.triangles = o3d.utility.Vector3iVector(target["triangles"])
        target_mesh.compute_vertex_normals()
        scene.add_geometry(
            "moving_candidate",
            target_mesh,
            _mat(target.get("color", _COLOR_CAND_SELECTED)),
        )

        camera = job["camera"]
        fov_type_name = str(camera.get("fov_type", "Vertical"))
        fov_type = getattr(
            rendering.Camera.FovType,
            fov_type_name,
            rendering.Camera.FovType.Vertical,
        )
        scene.camera.look_at(
            np.asarray(camera["center"], dtype=np.float64),
            np.asarray(camera["eye"], dtype=np.float64),
            np.asarray(camera["up"], dtype=np.float64),
        )
        scene.camera.set_projection(
            float(camera.get("fov", 60.0)),
            width / max(float(height), 1.0),
            float(camera.get("near", 0.01)),
            float(camera.get("far", 100.0)),
            fov_type,
        )

        frames = []
        for pose in np.asarray(job["poses"], dtype=np.float64):
            scene.set_geometry_transform(
                "moving_candidate",
                pose_to_transformation_matrix(pose[:7]),
            )
            frames.append(np.asarray(renderer.render_to_image()).copy())

        save_path = str(job["save_path"])
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        imageio.mimsave(save_path, frames, fps=float(job.get("fps", 30.0)))
        result_queue.put((True, save_path))
    except BaseException as exc:
        try:
            result_queue.put((False, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass


class CandidateViewer:
    def __init__(
        self,
        data_path: str,
        mesh_source: str = "auto",
        excavator: str = "auto",
    ):
        self._data_path = Path(data_path)
        self._data = _load_viewer_data(self._data_path)
        self._planning_params = self._load_planning_params()
        self._mesh_source = self._resolve_mesh_source(mesh_source)
        self._external_ply_meshes = {}
        self._external_dsf_meshes = {}
        self._asset_dir = self._resolve_asset_dir()
        self._target_structure_offset = self._resolve_target_structure_offset()
        self._sceneid_ground_height, self._sceneid_ground_plane = (
            self._resolve_sceneid_ground()
        )
        self._score_map_env = None
        self._score_map_cache = {}
        self._show_excavator = self._resolve_show_excavator(excavator)
        self._excavator_model = None
        self._excavator_meshes = {}
        self._excavator_q = self._initial_excavator_q()
        self._excavator_joint_edits = []
        self._excavator_q_label = None
        self._show_lidar_frames = False
        self._show_end_effector_frame = False
        self._syncing_excavator_controls = False
        if self._show_excavator:
            self._load_excavator()
        self._candidate_render_mode = "all"
        self._render_all_candidates = True
        self._candidate_pose_key = "solved_pose"
        self._show_score_map = True
        self._show_contacts = False
        self._show_pose_solve_contacts = False
        self._show_failed_grasps = False
        self._failed_grasp_index = -1  # -1 = all failed grasps
        self._gripper_model = None
        self._gripper_meshes = {}
        self._trajectory_video_fps = _DEFAULT_TRAJECTORY_VIDEO_FPS
        self._trajectory_video_exporting = False

        self._step_idx = 0
        self._cand_idx = 0
        self._fit_camera = True  # fit on step change; preserve on candidate change

        app = gui.Application.instance
        app.initialize()
        self._build_window()
        self._refresh_step_list()
        if self._data.get("steps"):
            self._select_step(len(self._data["steps"]) - 1)
        else:
            self._cand_list.set_items([])
            self._info_label.text = "No debug steps"
        app.run()

    def _load_planning_params(self) -> dict:
        params_path = self._data_path.parent / "planning_params.pkl"
        if not params_path.exists():
            return {}
        try:
            with params_path.open("rb") as f:
                params = pickle.load(f)
            return params if isinstance(params, dict) else {}
        except Exception:
            return {}

    def _resolve_show_excavator(self, mode: str) -> bool:
        mode = str(mode).lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(self._planning_params)

    def _resolve_mesh_source(self, mesh_source: str) -> str:
        if mesh_source != "auto":
            return mesh_source
        stored = str(self._data.get("mesh_source", "dsf"))
        if stored == "ply" and self._data.get("stone_ply_meshes"):
            return "ply"
        return stored if stored in {"dsf", "ply"} else "dsf"

    def _stone_mesh(self, stone_id: int):
        if self._mesh_source == "ply":
            ply_meshes = self._data.get("stone_ply_meshes", {})
            if stone_id in ply_meshes:
                return ply_meshes[stone_id]
            external = self._load_external_ply_mesh(stone_id)
            if external is not None:
                return external
        mesh = self._data.get("stone_meshes", {}).get(stone_id)
        if mesh is not None:
            return mesh
        return self._load_external_dsf_mesh(stone_id)

    def _resolve_target_structure_offset(self) -> np.ndarray:
        offset = self._data.get("target_structure_offset", None)
        if offset is None and self._planning_params:
            offset = self._planning_params.get("target_structure_offset", None)
        if offset is None:
            return np.zeros(3, dtype=float)
        arr = np.asarray(offset, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            return np.zeros(3, dtype=float)
        out = np.zeros(3, dtype=float)
        out[:2] = arr[:2]
        if arr.size >= 3 and np.isfinite(arr[2]):
            out[2] = arr[2]
        return out

    def _resolve_sceneid_ground(self) -> tuple[float | None, np.ndarray | None]:
        metadata = self._data.get("reconstructed_from_logs", {})
        if not isinstance(metadata, dict):
            return None, None

        ground_height = metadata.get("ground_height", None)
        try:
            ground_height = None if ground_height is None else float(ground_height)
        except (TypeError, ValueError):
            ground_height = None
        if ground_height is not None and not np.isfinite(ground_height):
            ground_height = None

        plane = metadata.get("ground_plane_model", None)
        if plane is not None:
            plane = np.asarray(plane, dtype=float).reshape(-1)
            if plane.shape != (4,) or not np.all(np.isfinite(plane)):
                plane = None

        return ground_height, plane

    def _display_pose(self, pose) -> np.ndarray:
        out = np.asarray(pose, dtype=float).copy()
        if out.ndim >= 1 and out.shape[0] >= 3:
            out[:3] += self._target_structure_offset
        return out

    def _display_points(self, points) -> np.ndarray:
        out = np.asarray(points, dtype=float).copy()
        if out.ndim >= 1 and out.shape[-1] >= 3:
            out[..., :3] += self._target_structure_offset
        return out

    def _ground_xy_bounds(self, wall_meshes: list, wall_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
        try:
            wall_width = float(wall_cfg.get("width", 2.0))
            wall_length = float(wall_cfg.get("length", 2.0))
        except (TypeError, ValueError):
            wall_width, wall_length = 2.0, 2.0

        half_extent = np.array(
            [
                max(wall_width, 0.0) * 0.5 + _GROUND_MARGIN,
                max(wall_length, 0.0) * 0.5 + _GROUND_MARGIN,
            ],
            dtype=float,
        )
        xy_min = -half_extent
        xy_max = half_extent

        for vertices, _triangles in wall_meshes:
            pts = np.asarray(vertices, dtype=float)
            if pts.ndim != 2 or pts.shape[1] < 2 or pts.size == 0:
                continue
            xy = pts[:, :2] + self._target_structure_offset[:2]
            finite = np.all(np.isfinite(xy), axis=1)
            if not np.any(finite):
                continue
            xy = xy[finite]
            xy_min = np.minimum(xy_min, xy.min(axis=0) - _GROUND_MARGIN)
            xy_max = np.maximum(xy_max, xy.max(axis=0) + _GROUND_MARGIN)

        return xy_min, xy_max

    def _origin_frame_size(self, wall_cfg: dict) -> float:
        """Axis-gizmo size at the world origin, proportional to target wall
        height (matches the original fixed size=0.4 at excavator-scale
        height=2.0, so it scales down with a smaller tabletop target)."""
        try:
            wall_height = float(wall_cfg.get("height", 2.0))
        except (TypeError, ValueError):
            wall_height = 2.0
        return max(0.01, 0.2 * wall_height)

    def _sceneid_ground_grid(
        self,
        xy_min: np.ndarray,
        xy_max: np.ndarray,
        n_lines: int = 12,
    ) -> o3d.geometry.LineSet | None:
        plane = self._sceneid_ground_plane
        if plane is None:
            return None
        a, b, c, d = plane
        if abs(float(c)) < 1e-9:
            return None

        xs = np.linspace(float(xy_min[0]), float(xy_max[0]), n_lines)
        ys = np.linspace(float(xy_min[1]), float(xy_max[1]), n_lines)
        points = []
        lines = []

        def point_at(x: float, y: float) -> np.ndarray:
            z = -(a * x + b * y + d) / c
            return np.array([x, y, z], dtype=float)

        for x in xs:
            i = len(points)
            points.extend([point_at(x, ys[0]), point_at(x, ys[-1])])
            lines.append([i, i + 1])
        for y in ys:
            i = len(points)
            points.extend([point_at(xs[0], y), point_at(xs[-1], y)])
            lines.append([i, i + 1])

        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        return grid

    def _sceneid_place_height_box(
        self,
        wall_meshes: list,
        pick_plane_height: float = 0.0,
    ) -> o3d.geometry.TriangleMesh | None:
        if self._sceneid_ground_height is None:
            return None

        points = []
        for vertices, _triangles in wall_meshes:
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim == 2 and vertices.shape[1] >= 3 and vertices.size > 0:
                points.append(vertices[:, :3] + self._target_structure_offset)
        if not points:
            return None

        pts = np.concatenate(points, axis=0)
        if not np.all(np.isfinite(pts)):
            return None

        xy_min = pts[:, :2].min(axis=0) - _PLACE_HEIGHT_BOX_MARGIN
        xy_max = pts[:, :2].max(axis=0) + _PLACE_HEIGHT_BOX_MARGIN
        extent_xy = xy_max - xy_min
        if np.any(extent_xy <= 0.0):
            return None

        z_top = float(self._sceneid_ground_height)
        thickness = max(
            abs(z_top - float(pick_plane_height)),
            _PLACE_HEIGHT_BOX_MIN_THICKNESS,
        )
        z_bottom = z_top - thickness

        mesh = o3d.geometry.TriangleMesh.create_box(
            float(extent_xy[0]), float(extent_xy[1]), float(thickness)
        )
        mesh.translate([float(xy_min[0]), float(xy_min[1]), float(z_bottom)])
        mesh.compute_vertex_normals()
        return mesh

    def _initial_excavator_q(self) -> np.ndarray:
        q = np.zeros(8, dtype=np.float64)
        q_source = self._planning_params.get("q_home", self._planner_q_preset("home"))
        q_source = np.asarray(q_source, dtype=np.float64).reshape(-1)
        if q_source.size >= 6 and np.all(np.isfinite(q_source[:6])):
            q[:6] = q_source[:6]
        return q

    @staticmethod
    def _planner_q_preset(name: str) -> np.ndarray:
        try:
            from planning import Q_HOME, Q_SCAN, Q_SCAN_INHAND

            presets = {
                "home": Q_HOME,
                "scan": Q_SCAN,
                "inhand": Q_SCAN_INHAND,
            }
            return np.asarray(presets[name], dtype=np.float64).copy()
        except Exception:
            return np.zeros(6, dtype=np.float64)

    def _load_excavator(self) -> None:
        try:
            from model import get_excavator_model

            self._excavator_model, self._excavator_meshes = get_excavator_model()
        except Exception as exc:
            print(f"[WARN] Could not load excavator model: {exc}")
            self._show_excavator = False
            self._excavator_model = None
            self._excavator_meshes = {}

    def _load_gripper_model(self) -> None:
        if self._gripper_model is not None:
            return
        try:
            from model import get_gripper_model

            self._gripper_model, self._gripper_meshes = get_gripper_model()
        except Exception as exc:
            print(f"[WARN] Could not load gripper model: {exc}")
            self._gripper_model = None
            self._gripper_meshes = {}

    def _excavator_mesh_entries(self):
        if not self._show_excavator or self._excavator_model is None:
            return []
        try:
            from model import update_urdf_mesh

            meshes = update_urdf_mesh(
                self._excavator_model,
                self._excavator_meshes,
                self._excavator_q.copy(),
            )
        except Exception:
            return []
        return [
            (f"excavator_{name}", mesh, _mat(_COLOR_EXCAVATOR))
            for name, mesh in meshes.items()
        ]

    def _excavator_frame_entries(self):
        if not self._show_excavator or self._excavator_model is None:
            return []
        try:
            self._excavator_model.SetState(self._excavator_q.copy())
        except Exception:
            return []

        entries = []
        if self._show_lidar_frames:
            for link_name in _LIDAR_FRAME_LINKS:
                frame = self._link_frame(link_name, size=0.45)
                if frame is not None:
                    entries.append((f"frame_{link_name}", frame, _mat([1, 1, 1, 1])))
        if self._show_end_effector_frame:
            frame = self._link_frame(_END_EFFECTOR_LINK, size=0.6, use_geom=True)
            if frame is not None:
                entries.append(("frame_end_effector", frame, _mat([1, 1, 1, 1])))
        return entries

    def _link_frame(self, link_name: str, size: float = 0.5, use_geom: bool = False):
        try:
            link = self._excavator_model.GetLink(link_name)
            source = link.geoms[0] if use_geom else link
            pose = np.eye(4)
            pose[:3, :3] = source.GetRotation()
            pose[:3, 3] = source.GetPosition()
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            frame.transform(pose)
            frame.compute_vertex_normals()
            return frame
        except Exception:
            return None

    def _resolve_asset_dir(self) -> Path | None:
        stored = self._data.get("asset_dir", None)
        if stored:
            path = Path(str(stored))
            return path if path.is_absolute() else Path(REPO_ROOT) / path
        config_path = self._data_path.parent / "config.yml"
        if not config_path.exists():
            return None
        cfg = OmegaConf.load(config_path)
        load_dir = OmegaConf.select(cfg, "environment.data.load_dir")
        if load_dir is None:
            load_dir = OmegaConf.select(cfg, "data.load_dir")
        if load_dir is None:
            return None
        path = Path(str(load_dir))
        if not path.is_absolute():
            path = Path(REPO_ROOT) / path
        return path

    def _load_external_ply_mesh(self, stone_id: int):
        if stone_id in self._external_ply_meshes:
            return self._external_ply_meshes[stone_id]
        if self._asset_dir is None:
            return None

        path = self._asset_dir / f"model_{stone_id}_mesh.ply"
        if not path.exists():
            return None
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            return None
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh_data = (
            np.asarray(mesh.vertices, dtype=float).copy(),
            np.asarray(mesh.triangles, dtype=int).copy(),
        )
        self._external_ply_meshes[stone_id] = mesh_data
        return mesh_data

    def _stone_asset_path(self, stone_id: int, suffix: str) -> Path | None:
        if self._asset_dir is None:
            return None
        direct = self._asset_dir / f"model_{stone_id}{suffix}"
        if direct.exists():
            return direct
        matches = sorted(self._asset_dir.rglob(f"model_{stone_id}{suffix}"))
        return matches[0] if matches else None

    def _load_external_dsf_mesh(self, stone_id: int):
        if stone_id in self._external_dsf_meshes:
            return self._external_dsf_meshes[stone_id]

        obj_path = self._stone_asset_path(stone_id, ".obj")
        if obj_path is None:
            return None

        try:
            importer = WavefrontImporter(str(obj_path))
            vertices_list, triangles_list, offset = [], [], 0
            for dsf_obj in importer.get_objects():
                dsf = DiffSupportSimple(
                    vertex_set=dsf_obj.vertices.T,
                    sharpness=dsf_obj.sharpness,
                )
                vertices, triangles = dsf.get_mesh(resolution=4)
                vertices_list.append(vertices)
                triangles_list.append(triangles + offset)
                offset += vertices.shape[0]
            if not vertices_list:
                return None

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(
                np.concatenate(vertices_list, axis=0)
            )
            mesh.triangles = o3d.utility.Vector3iVector(
                np.concatenate(triangles_list, axis=0)
            )
            mesh.orient_triangles()
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            mesh = fix_normals_outward(mesh)
        except Exception:
            return None

        mesh_data = (
            np.asarray(mesh.vertices, dtype=float).copy(),
            np.asarray(mesh.triangles, dtype=int).copy(),
        )
        self._external_dsf_meshes[stone_id] = mesh_data
        return mesh_data

    def _add_mesh(self, name: str, mesh, mat) -> bool:
        return self._add_geometry(name, mesh, mat)

    def _add_geometry(self, name: str, geom, mat) -> bool:
        if not _geometry_renderable(geom):
            return False
        try:
            self._sw.scene.add_geometry(name, geom, mat)
            return True
        except RuntimeError:
            return False

    @staticmethod
    def _floor_cell_mesh(x0: float, x1: float, y0: float, y1: float, z: float):
        mesh = o3d.geometry.TriangleMesh.create_box(
            max(float(x1 - x0), 1e-4),
            max(float(y1 - y0), 1e-4),
            0.003,
        )
        mesh.translate([float(x0), float(y0), float(z)])
        mesh.compute_vertex_normals()
        return mesh

    def _add_floor_fill_overlay(self, floor_data: dict | None) -> None:
        if not isinstance(floor_data, dict):
            return
        occupied = np.asarray(floor_data.get("occupied", []), dtype=bool)
        xs = np.asarray(floor_data.get("x_coords", []), dtype=float).reshape(-1)
        ys = np.asarray(floor_data.get("y_coords", []), dtype=float).reshape(-1)
        if occupied.ndim != 2 or len(xs) < 2 or len(ys) < 2:
            return
        if occupied.shape != (len(ys), len(xs)):
            return

        offset = self._target_structure_offset
        xs = xs + float(offset[0])
        ys = ys + float(offset[1])
        dx = float(np.median(np.diff(xs)))
        dy = float(np.median(np.diff(ys)))
        if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
            return
        z = float(floor_data.get("support_z", 0.0)) + float(offset[2]) + 0.006
        mat = _mat(_COLOR_ACTIVE_FLOOR_OCCUPIED, transparent=True)
        for iy, ix in np.argwhere(occupied):
            x0 = float(xs[ix] - 0.5 * dx)
            x1 = float(xs[ix] + 0.5 * dx)
            y0 = float(ys[iy] - 0.5 * dy)
            y1 = float(ys[iy] + 0.5 * dy)
            mesh = self._floor_cell_mesh(x0, x1, y0, y1, z)
            self._add_mesh(f"floor_fill_occ_{iy}_{ix}", mesh, mat)

    # ------------------------------------------------------------------
    # Window / layout
    # ------------------------------------------------------------------
    def _build_window(self):
        self._win = gui.Application.instance.create_window(
            "MCTS Candidate Viewer", 1400, 800
        )
        em = self._win.theme.font_size
        margin  = int(0.50 * em)
        spacing = int(0.40 * em)

        # 3-D scene widget (right side, flexible)
        self._sw = gui.SceneWidget()
        self._sw.scene = rendering.Open3DScene(self._win.renderer)
        self._sw.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self._sw.scene.scene.set_indirect_light_intensity(30000)
        self._win.add_child(self._sw)

        # Left control panel
        self._panel = gui.Vert(
            spacing, gui.Margins(margin, margin, margin, margin)
        )

        self._panel.add_child(gui.Label("Mesh"))
        self._mesh_combo = gui.Combobox()
        self._mesh_combo.add_item("DSF")
        self._mesh_combo.add_item("PLY")
        self._mesh_combo.selected_index = 1 if self._mesh_source == "ply" else 0
        self._mesh_combo.set_on_selection_changed(self._on_mesh_source_changed)
        self._panel.add_child(self._mesh_combo)
        self._panel.add_fixed(spacing)

        self._panel.add_child(gui.Label("Candidate Render"))
        self._candidate_render_combo = gui.Combobox()
        self._candidate_render_combo.add_item("All")
        self._candidate_render_combo.add_item("All opaque")
        self._candidate_render_combo.add_item("Selected")
        self._candidate_render_combo.add_item("No select")
        self._candidate_render_combo.selected_index = 0
        self._candidate_render_combo.set_on_selection_changed(
            self._on_candidate_render_changed
        )
        self._panel.add_child(self._candidate_render_combo)
        self._panel.add_fixed(spacing)

        self._panel.add_child(gui.Label("Candidate Pose"))
        self._candidate_pose_combo = gui.Combobox()
        self._candidate_pose_combo.add_item("Solved")
        self._candidate_pose_combo.add_item("Settled")
        self._candidate_pose_combo.add_item("Initial")
        self._candidate_pose_combo.add_item("Initial + Solved")
        self._candidate_pose_combo.add_item("Final")
        self._candidate_pose_combo.add_item("Trajectory")
        self._candidate_pose_combo.add_item("Best sequence")
        self._candidate_pose_combo.selected_index = 0
        self._candidate_pose_combo.set_on_selection_changed(
            self._on_candidate_pose_changed
        )
        self._panel.add_child(self._candidate_pose_combo)
        self._panel.add_fixed(spacing)

        video_row = gui.Horiz(spacing)
        video_row.add_child(gui.Label("Video FPS"))
        self._trajectory_video_fps_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._trajectory_video_fps_edit.set_limits(1.0, 60.0)
        self._trajectory_video_fps_edit.double_value = self._trajectory_video_fps
        self._trajectory_video_fps_edit.set_on_value_changed(
            self._on_trajectory_video_fps_changed
        )
        video_row.add_child(self._trajectory_video_fps_edit)
        self._panel.add_child(video_row)

        self._save_trajectory_video_btn = gui.Button("Save trajectory MP4")
        self._save_trajectory_video_btn.set_on_clicked(
            self._on_save_trajectory_video_clicked
        )
        self._panel.add_child(self._save_trajectory_video_btn)
        self._trajectory_video_status_label = gui.Label("")
        self._panel.add_child(self._trajectory_video_status_label)
        self._panel.add_fixed(spacing)

        self._show_score_map_cb = gui.Checkbox("Show state score map")
        self._show_score_map_cb.checked = self._show_score_map
        self._show_score_map_cb.set_on_checked(self._on_show_score_map_changed)
        self._panel.add_child(self._show_score_map_cb)
        self._panel.add_fixed(spacing)

        self._show_contacts_cb = gui.Checkbox("Show contacts")
        self._show_contacts_cb.checked = self._show_contacts
        self._show_contacts_cb.set_on_checked(self._on_show_contacts_changed)
        self._panel.add_child(self._show_contacts_cb)
        self._panel.add_fixed(spacing)

        self._show_pose_solve_contacts_cb = gui.Checkbox("Show pose solve forces")
        self._show_pose_solve_contacts_cb.checked = self._show_pose_solve_contacts
        self._show_pose_solve_contacts_cb.set_on_checked(
            self._on_show_pose_solve_contacts_changed
        )
        self._panel.add_child(self._show_pose_solve_contacts_cb)
        self._panel.add_fixed(spacing)

        self._show_failed_grasps_cb = gui.Checkbox("Show failed grasps")
        self._show_failed_grasps_cb.checked = self._show_failed_grasps
        self._show_failed_grasps_cb.set_on_checked(
            self._on_show_failed_grasps_changed
        )
        self._panel.add_child(self._show_failed_grasps_cb)

        failed_grasp_row = gui.Horiz(spacing)
        failed_grasp_row.add_child(gui.Label("Grasp idx (-1=all)"))
        self._failed_grasp_idx_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._failed_grasp_idx_edit.set_limits(-1, 9999)
        self._failed_grasp_idx_edit.int_value = self._failed_grasp_index
        self._failed_grasp_idx_edit.set_on_value_changed(
            self._on_failed_grasp_idx_changed
        )
        failed_grasp_row.add_child(self._failed_grasp_idx_edit)
        self._panel.add_child(failed_grasp_row)
        self._panel.add_fixed(spacing)

        if self._show_excavator:
            self._add_excavator_controls(spacing)

        self._refresh_btn = gui.Button("Refresh")
        self._refresh_btn.set_on_clicked(self._on_refresh_clicked)
        self._panel.add_child(self._refresh_btn)
        self._panel.add_fixed(spacing)

        self._panel.add_child(gui.Label("Steps"))
        self._step_list = gui.ListView()
        self._step_list.set_max_visible_items(8)
        self._step_list.set_on_selection_changed(self._on_step_changed)
        self._panel.add_child(self._step_list)

        self._panel.add_fixed(spacing)
        self._panel.add_child(gui.Label("Candidates"))
        self._cand_list = gui.ListView()
        self._cand_list.set_max_visible_items(24)
        self._cand_list.set_on_selection_changed(self._on_cand_changed)
        self._panel.add_child(self._cand_list)

        self._panel.add_fixed(spacing)
        self._info_label = gui.Label("—")
        self._panel.add_child(self._info_label)

        self._win.add_child(self._panel)
        self._win.set_on_layout(self._on_layout)

    def _add_excavator_controls(self, spacing: int) -> None:
        self._panel.add_child(gui.Label("Excavator"))

        preset_row = gui.Horiz(spacing)
        for label, q in (
            ("Zero", np.zeros(8, dtype=float)),
            ("Home", self._q8_from_q6(self._planner_q_preset("home"))),
            ("Scan", self._q8_from_q6(self._planner_q_preset("scan"))),
            ("Inhand", self._q8_from_q6(self._planner_q_preset("inhand"))),
        ):
            button = gui.Button(label)
            button.set_on_clicked(lambda q=q: self._set_excavator_q(q))
            preset_row.add_child(button)
        self._panel.add_child(preset_row)

        frame_row = gui.Horiz(spacing)
        self._lidar_frames_cb = gui.Checkbox("LiDAR frames")
        self._lidar_frames_cb.checked = self._show_lidar_frames
        self._lidar_frames_cb.set_on_checked(self._on_lidar_frames_changed)
        self._ee_frame_cb = gui.Checkbox("EE frame")
        self._ee_frame_cb.checked = self._show_end_effector_frame
        self._ee_frame_cb.set_on_checked(self._on_end_effector_frame_changed)
        frame_row.add_child(self._lidar_frames_cb)
        frame_row.add_child(self._ee_frame_cb)
        self._panel.add_child(frame_row)

        self._excavator_joint_edits = []
        for name, idx, min_deg, max_deg in _EXCAVATOR_JOINTS:
            row = gui.Horiz(spacing)
            row.add_child(gui.Label(name))
            edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            edit.set_limits(float(min_deg), float(max_deg))
            edit.double_value = float(np.rad2deg(self._excavator_q[idx]))
            edit.set_on_value_changed(
                lambda _value, joint_idx=idx: self._on_excavator_joint_changed(
                    joint_idx
                )
            )
            self._excavator_joint_edits.append((idx, edit))
            row.add_child(edit)
            row.add_child(gui.Label("deg"))
            self._panel.add_child(row)

        self._excavator_q_label = gui.Label("")
        self._panel.add_child(self._excavator_q_label)
        self._panel.add_fixed(spacing)
        self._sync_excavator_controls_from_q()

    def _on_layout(self, _ctx):
        r = self._win.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, _PANEL_W, r.height)
        self._sw.frame = gui.Rect(
            r.x + _PANEL_W, r.y, r.width - _PANEL_W, r.height
        )

    @staticmethod
    def _q8_from_q6(q6) -> np.ndarray:
        q = np.zeros(8, dtype=np.float64)
        q6 = np.asarray(q6, dtype=np.float64).reshape(-1)
        if q6.size >= 6:
            q[:6] = q6[:6]
        return q

    def _set_excavator_q(self, q, redraw: bool = True) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size < 8 or not np.all(np.isfinite(q[:8])):
            return
        self._excavator_q = q[:8].copy()
        self._sync_excavator_controls_from_q()
        if redraw:
            self._fit_camera = False
            self._render()

    def _sync_excavator_controls_from_q(self) -> None:
        self._syncing_excavator_controls = True
        try:
            for idx, edit in self._excavator_joint_edits:
                edit.double_value = float(np.rad2deg(self._excavator_q[idx]))
        finally:
            self._syncing_excavator_controls = False
        if self._excavator_q_label is not None:
            self._excavator_q_label.text = (
                "q rad: "
                + np.array2string(
                    self._excavator_q,
                    precision=3,
                    suppress_small=True,
                )
            )

    def _on_excavator_joint_changed(self, joint_idx: int) -> None:
        if self._syncing_excavator_controls:
            return
        for idx, edit in self._excavator_joint_edits:
            if idx != joint_idx:
                continue
            self._excavator_q[idx] = np.deg2rad(float(edit.double_value))
            break
        self._sync_excavator_controls_from_q()
        self._fit_camera = False
        self._render()

    def _on_lidar_frames_changed(self, checked: bool) -> None:
        self._show_lidar_frames = bool(checked)
        self._fit_camera = False
        self._render()

    def _on_end_effector_frame_changed(self, checked: bool) -> None:
        self._show_end_effector_frame = bool(checked)
        self._fit_camera = False
        self._render()

    # ------------------------------------------------------------------
    # List helpers
    # ------------------------------------------------------------------
    def _step_label(self, s: dict) -> str:
        if bool(s.get("state_only", False)):
            return f"Step {s['step']:2d}   (state checkpoint)"
        reason = s.get("failure_reason")
        if reason == "motion_planning_failed":
            flag = " [MOTION FAIL]"
        else:
            flag = " [NO CAND]" if not s.get("succeeded", True) else ""
        attempt = s.get("attempt", None)
        attempt_label = f" attempt {attempt}" if attempt is not None else ""
        rejected_stones = s.get("rejected_stone_ids", []) or []
        rejected_label = (
            f", {len(rejected_stones)} rejected stones" if rejected_stones else ""
        )
        return (
            f"Step {s['step']:2d}{attempt_label}   "
            f"({len(s['candidates'])} candidates{rejected_label}){flag}"
        )

    def _cand_label(self, c: dict) -> str:
        if not self._candidate_validated(c):
            flag = " ?"
        elif c.get("motion_failed", False):
            flag = " motion"
        elif (c.get("info", {}) or {}).get("final_validation_failed", False):
            flag = " final"
        elif c["failed"]:
            flag = " failed"
        else:
            flag = ""
        selected = " *" if c.get("selected", False) else ""
        return (
            f"#{c['rank']:2d}  {c['score']:+.4f}  "
            f"stone {c['stone_id']}{flag}{selected}"
        )

    @staticmethod
    def _candidate_validated(c: dict) -> bool:
        if "validated" in c:
            return bool(c.get("validated", False))
        return c.get("reward") is not None or int(c.get("visits", 0)) > 0

    @staticmethod
    def _selected_candidate_idx(step: dict) -> int:
        for i, candidate in enumerate(step.get("candidates", [])):
            if candidate.get("selected", False):
                return i
        selected = step.get("selected_candidate") or {}
        selected_rank = selected.get("rank")
        if selected_rank is None:
            return 0
        for i, candidate in enumerate(step.get("candidates", [])):
            if int(candidate.get("rank", -1)) == int(selected_rank):
                return i
        return 0

    def _refresh_step_list(self):
        self._step_list.set_items(
            [self._step_label(s) for s in self._data["steps"]]
        )

    def _refresh_cand_list(self):
        step = self._data["steps"][self._step_idx]
        self._cand_list.set_items(
            [self._cand_label(c) for c in step["candidates"]]
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_step_changed(self, item: str, _dbl: bool):
        for i, s in enumerate(self._data["steps"]):
            if self._step_label(s) == item:
                self._fit_camera = True
                self._select_step(i)
                return

    def _on_cand_changed(self, item: str, _dbl: bool):
        step = self._data["steps"][self._step_idx]
        for i, c in enumerate(step["candidates"]):
            if self._cand_label(c) == item:
                self._cand_idx = i
                self._fit_camera = False
                self._render()
                return

    def _on_mesh_source_changed(self, item: str, _idx: int):
        self._mesh_source = "ply" if item.lower() == "ply" else "dsf"
        self._fit_camera = False
        self._render()

    def _on_candidate_render_changed(self, item: str, _idx: int):
        text = item.lower()
        if text == "selected":
            self._candidate_render_mode = "selected"
        elif text == "all opaque":
            self._candidate_render_mode = "all_opaque"
        elif text == "no select":
            self._candidate_render_mode = "no_select"
        else:
            self._candidate_render_mode = "all"
        self._render_all_candidates = self._candidate_render_mode != "selected"
        self._fit_camera = False
        self._render()

    def _on_candidate_pose_changed(self, item: str, _idx: int):
        text = item.lower()
        if text == "solved":
            self._candidate_pose_key = "solved_pose"
        elif text == "settled":
            self._candidate_pose_key = "settled_pose"
        elif text == "initial":
            self._candidate_pose_key = "init_pose"
        elif text == "initial + solved":
            self._candidate_pose_key = "init_and_solved"
        elif text == "final":
            self._candidate_pose_key = "final_pose"
        elif text == "trajectory":
            self._candidate_pose_key = "trajectory"
        elif text == "best sequence":
            self._candidate_pose_key = "best_sequence"
        else:
            self._candidate_pose_key = "solved_pose"
        self._fit_camera = False
        self._render()

    def _on_trajectory_video_fps_changed(self, _value):
        self._trajectory_video_fps = float(
            np.clip(self._trajectory_video_fps_edit.double_value, 1.0, 60.0)
        )

    def _on_save_trajectory_video_clicked(self):
        if self._trajectory_video_exporting:
            return
        try:
            job = self._build_trajectory_video_job()
        except Exception as exc:
            self._set_trajectory_video_status(f"Video setup failed: {exc}", False)
            return
        if job is None:
            return

        self._set_trajectory_video_status(
            f"Saving {Path(job['save_path']).name}...", True
        )
        try:
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            process = ctx.Process(
                target=_candidate_trajectory_video_process,
                args=(job, result_queue),
                daemon=True,
            )
            process.start()
        except Exception as exc:
            self._set_trajectory_video_status(f"Video start failed: {exc}", False)
            return

        watcher = threading.Thread(
            target=self._watch_trajectory_video_process,
            args=(process, result_queue, str(job["save_path"])),
            daemon=True,
        )
        watcher.start()

    def _watch_trajectory_video_process(self, process, result_queue, save_path: str):
        process.join()
        ok = False
        message = ""
        try:
            ok, message = result_queue.get(timeout=1.0)
        except queue.Empty:
            if process.exitcode == 0:
                ok = True
                message = save_path
            else:
                message = f"renderer exited with code {process.exitcode}"

        def finish():
            if ok:
                self._set_trajectory_video_status(f"Saved {message}", False)
            else:
                self._set_trajectory_video_status(f"Video failed: {message}", False)

        gui.Application.instance.post_to_main_thread(self._win, finish)

    def _set_trajectory_video_status(self, text: str, exporting: bool):
        self._trajectory_video_exporting = bool(exporting)
        self._save_trajectory_video_btn.enabled = not self._trajectory_video_exporting
        self._trajectory_video_status_label.text = str(text)
        self._win.post_redraw()

    def _on_show_score_map_changed(self, checked: bool):
        self._show_score_map = bool(checked)
        self._fit_camera = False
        self._render()

    def _on_show_contacts_changed(self, checked: bool):
        self._show_contacts = bool(checked)
        self._fit_camera = False
        self._render()

    def _on_show_pose_solve_contacts_changed(self, checked: bool):
        self._show_pose_solve_contacts = bool(checked)
        self._fit_camera = False
        self._render()

    def _on_show_failed_grasps_changed(self, checked: bool):
        self._show_failed_grasps = bool(checked)
        if self._show_failed_grasps:
            self._load_gripper_model()
        self._fit_camera = False
        self._render()

    def _on_failed_grasp_idx_changed(self, value):
        self._failed_grasp_index = int(value)
        self._fit_camera = False
        self._render()

    def _on_refresh_clicked(self):
        self._reload_data()

    def _reload_data(self):
        try:
            new_data = _load_viewer_data(self._data_path)
        except Exception:
            return
        self._data = new_data

        prev_step = self._step_idx
        self._refresh_step_list()

        n = len(self._data["steps"])
        if n == 0:
            return
        new_idx = min(prev_step, n - 1)
        if new_idx != self._step_idx or n > prev_step + 1:
            self._fit_camera = True
        self._step_idx = new_idx
        self._cand_idx = self._selected_candidate_idx(self._data["steps"][new_idx])
        self._refresh_cand_list()
        self._render()

    def _select_step(self, idx: int):
        self._step_idx = idx
        self._cand_idx = self._selected_candidate_idx(self._data["steps"][idx])
        self._refresh_cand_list()
        self._render()

    # ------------------------------------------------------------------
    # Trajectory MP4 export
    # ------------------------------------------------------------------
    def _build_trajectory_video_job(self) -> dict | None:
        if not self._data.get("steps"):
            self._set_trajectory_video_status("No debug step loaded", False)
            return None
        step = self._data["steps"][self._step_idx]
        candidates = step.get("candidates", []) or []
        if self._cand_idx >= len(candidates):
            self._set_trajectory_video_status("No candidate selected", False)
            return None

        candidate = candidates[self._cand_idx]
        poses = self._candidate_trajectory_poses(candidate)
        if len(poses) == 0:
            self._set_trajectory_video_status(
                "Selected candidate has no trajectory", False
            )
            return None

        stone_id = int(candidate.get("stone_id", -1))
        mesh_data = self._stone_mesh(stone_id)
        if mesh_data is None:
            self._set_trajectory_video_status(
                "Selected stone mesh is unavailable", False
            )
            return None

        width, height = self._trajectory_video_dimensions()
        camera = self._current_camera_params(width, height)
        save_path = self._trajectory_video_path(step, candidate)
        static_entries = self._trajectory_video_static_entries(step, poses)
        vertices, triangles = mesh_data
        return {
            "width": width,
            "height": height,
            "fps": float(self._trajectory_video_fps),
            "save_path": str(save_path),
            "camera": camera,
            "poses": poses.astype(np.float64).copy(),
            "target_mesh": {
                "vertices": np.asarray(vertices, dtype=np.float64).copy(),
                "triangles": np.asarray(triangles, dtype=np.int32).copy(),
                "color": list(_COLOR_CAND_SELECTED),
            },
            "static_entries": static_entries,
        }

    def _candidate_trajectory_poses(self, candidate: dict) -> np.ndarray:
        raw = np.asarray(candidate.get("trajectory", []), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 7:
            return np.empty((0, 7), dtype=np.float64)
        raw = raw[np.all(np.isfinite(raw[:, :7]), axis=1), :7]
        if len(raw) == 0:
            return np.empty((0, 7), dtype=np.float64)
        return np.asarray(
            [self._display_pose(pose)[:7] for pose in raw],
            dtype=np.float64,
        )

    def _trajectory_video_dimensions(self) -> tuple[int, int]:
        try:
            width = int(self._sw.frame.width)
            height = int(self._sw.frame.height)
        except Exception:
            width, height = 1280, 720
        if width < 320 or height < 240:
            width, height = 1280, 720
        width -= width % 2
        height -= height % 2
        return max(width, 2), max(height, 2)

    def _current_camera_params(self, width: int, height: int) -> dict:
        camera = self._sw.scene.camera
        model = self._camera_model_matrix(camera)
        eye = model[:3, 3].astype(float)
        forward = -model[:3, 2].astype(float)
        up = model[:3, 1].astype(float)
        forward = self._normalized_or(forward, np.array([0.0, 0.0, -1.0]))
        up = self._normalized_or(up, np.array([0.0, 0.0, 1.0]))
        center = eye + forward

        try:
            fov = float(camera.get_field_of_view())
        except Exception:
            fov = 60.0
        if not np.isfinite(fov) or fov <= 0.0:
            fov = 60.0
        try:
            fov_type = camera.get_field_of_view_type().name
        except Exception:
            fov_type = "Vertical"

        extent = self._current_scene_extent()
        return {
            "eye": eye.tolist(),
            "center": center.tolist(),
            "up": up.tolist(),
            "fov": fov,
            "fov_type": fov_type,
            "near": max(extent * 1e-4, 0.01),
            "far": max(extent * 20.0, 100.0),
            "width": int(width),
            "height": int(height),
        }

    @staticmethod
    def _camera_model_matrix(camera) -> np.ndarray:
        try:
            matrix = np.asarray(camera.get_model_matrix(), dtype=np.float64).reshape(
                4, 4
            )
            if np.all(np.isfinite(matrix)):
                return matrix
        except Exception:
            pass
        try:
            view = np.asarray(camera.get_view_matrix(), dtype=np.float64).reshape(4, 4)
            if np.all(np.isfinite(view)):
                return np.linalg.inv(view)
        except Exception:
            pass
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.array([4.0, 4.0, 3.0])
        return matrix

    @staticmethod
    def _normalized_or(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if np.isfinite(norm) and norm > 1e-9:
            return vector / norm
        return np.asarray(fallback, dtype=float).reshape(3)

    def _current_scene_extent(self) -> float:
        try:
            bounds = self._sw.scene.bounding_box
            extent = np.asarray(bounds.get_extent(), dtype=float).reshape(3)
            value = float(np.linalg.norm(extent))
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        return 10.0

    def _trajectory_video_path(self, step: dict, candidate: dict) -> Path:
        out_dir = self._data_path.parent / _TRAJECTORY_VIDEO_SUBDIR
        step_num = int(step.get("step", self._step_idx + 1))
        rank = int(candidate.get("rank", self._cand_idx + 1))
        stone_id = int(candidate.get("stone_id", -1))
        return out_dir / f"step_{step_num:02d}_rank_{rank:02d}_stone_{stone_id}.mp4"

    def _trajectory_video_static_entries(
        self,
        step: dict,
        poses: np.ndarray,
    ) -> list[dict]:
        wall_meshes: list = self._data["target_wall_meshes"]
        wall_cfg: dict = self._data.get("target_wall_cfg", {})
        entries: list[dict] = []

        ground_xy_min, ground_xy_max = self._ground_xy_bounds(wall_meshes, wall_cfg)
        ground_extent = ground_xy_max - ground_xy_min
        ground_z = (
            float(self._sceneid_ground_height)
            if self._sceneid_ground_height is not None
            else 0.0
        )
        ground_color = (
            _COLOR_SCENEID_GROUND
            if self._sceneid_ground_height is not None
            else _COLOR_GROUND
        )
        ground = o3d.geometry.TriangleMesh.create_box(
            float(ground_extent[0]), float(ground_extent[1]), _GROUND_DEPTH
        )
        ground.translate(
            [
                float(ground_xy_min[0]),
                float(ground_xy_min[1]),
                ground_z - _GROUND_DEPTH,
            ]
        )
        ground.compute_vertex_normals()
        self._append_video_mesh(
            entries,
            "ground",
            ground,
            ground_color,
            transparent=self._sceneid_ground_height is not None,
        )

        grid = self._sceneid_ground_grid(ground_xy_min, ground_xy_max)
        self._append_video_line(
            entries,
            "sceneid_ground_grid",
            grid,
            _COLOR_SCENEID_GROUND,
            width=2.0,
        )

        place_box = self._sceneid_place_height_box(wall_meshes)
        self._append_video_mesh(
            entries,
            "sceneid_place_height_box",
            place_box,
            _COLOR_SCENEID_PLACE_BOX,
            transparent=True,
        )

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self._origin_frame_size(wall_cfg)
        )
        frame.compute_vertex_normals()
        self._append_video_mesh(entries, "coord_frame", frame, [1.0, 1.0, 1.0, 1.0])

        for i, (v, t) in enumerate(wall_meshes):
            wall_mesh = _make_mesh(v, t)
            wall_mesh.translate(self._target_structure_offset)
            self._append_video_mesh(
                entries,
                f"wall_{i}",
                wall_mesh,
                _COLOR_WALL,
                transparent=True,
            )

        scene = step.get("scene", {}) or {}
        scene_poses = scene.get("stone_poses", {}) or {}
        for stone_id in scene.get("stone_seq", []) or []:
            sid = int(stone_id)
            mesh_data = self._stone_mesh(sid)
            pose = self._display_pose(
                scene_poses.get(sid, scene_poses.get(str(sid), []))
            )
            if (
                mesh_data is None
                or pose.shape[0] < 7
                or not np.all(np.isfinite(pose[:7]))
            ):
                continue
            mesh = _make_mesh(*mesh_data)
            mesh.transform(pose_to_transformation_matrix(pose[:7]))
            self._append_video_mesh(entries, f"scene_{sid}", mesh, _COLOR_SCENE_STONE)

        for entry in self._trajectory_path_video_entries(poses):
            entries.append(entry)
        return entries

    def _trajectory_path_video_entries(self, poses: np.ndarray) -> list[dict]:
        points = np.asarray(poses[:, :3], dtype=float)
        entries: list[dict] = []
        if len(points) >= 2:
            max_points = 256
            if len(points) > max_points:
                points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
            lines = np.column_stack(
                [np.arange(len(points) - 1), np.arange(1, len(points))]
            )
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points.astype(float))
            line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
            entry = _video_line_entry(
                "candidate_trajectory_path",
                line_set,
                _COLOR_TRAJ_LINE,
                width=4.0,
            )
            if entry is not None:
                entries.append(entry)

        for name, point, color in (
            ("candidate_trajectory_start", points[0], _COLOR_TRAJ_START),
            ("candidate_trajectory_end", points[-1], _COLOR_TRAJ_END),
        ):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
            sphere.translate(point)
            sphere.compute_vertex_normals()
            entry = _video_mesh_entry(name, sphere, color)
            if entry is not None:
                entries.append(entry)
        return entries

    @staticmethod
    def _append_video_mesh(
        entries: list[dict],
        name: str,
        mesh,
        color: list,
        transparent: bool = False,
    ) -> None:
        if mesh is None:
            return
        entry = _video_mesh_entry(name, mesh, color, transparent=transparent)
        if entry is not None:
            entries.append(entry)

    @staticmethod
    def _append_video_line(
        entries: list[dict],
        name: str,
        line_set,
        color: list,
        width: float = 4.0,
    ) -> None:
        if line_set is None:
            return
        entry = _video_line_entry(name, line_set, color, width=width)
        if entry is not None:
            entries.append(entry)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self):
        sc = self._sw.scene
        sc.clear_geometry()

        data        = self._data
        step        = data["steps"][self._step_idx]
        wall_meshes: list  = data["target_wall_meshes"] # list of (verts, tris)
        wall_cfg: dict     = data.get("target_wall_cfg", {})

        # Ground plane
        ground_xy_min, ground_xy_max = self._ground_xy_bounds(wall_meshes, wall_cfg)
        ground_extent = ground_xy_max - ground_xy_min
        ground_z = (
            float(self._sceneid_ground_height)
            if self._sceneid_ground_height is not None
            else 0.0
        )
        ground_color = (
            _COLOR_SCENEID_GROUND
            if self._sceneid_ground_height is not None
            else _COLOR_GROUND
        )
        ground = o3d.geometry.TriangleMesh.create_box(
            float(ground_extent[0]), float(ground_extent[1]), _GROUND_DEPTH
        )
        ground.translate(
            [
                float(ground_xy_min[0]),
                float(ground_xy_min[1]),
                ground_z - _GROUND_DEPTH,
            ]
        )
        ground.compute_vertex_normals()
        self._add_mesh(
            "ground",
            ground,
            _mat(ground_color, transparent=self._sceneid_ground_height is not None),
        )
        sceneid_ground_grid = self._sceneid_ground_grid(ground_xy_min, ground_xy_max)
        if sceneid_ground_grid is not None:
            self._add_geometry(
                "sceneid_ground_grid",
                sceneid_ground_grid,
                _line_mat(_COLOR_SCENEID_GROUND, width=2.0),
            )
        sceneid_place_box = self._sceneid_place_height_box(wall_meshes)
        if sceneid_place_box is not None:
            self._add_mesh(
                "sceneid_place_height_box",
                sceneid_place_box,
                _mat(_COLOR_SCENEID_PLACE_BOX, transparent=True),
            )
        self._add_floor_fill_overlay(step.get("floor_fill"))
        score_map = self._score_map_for_step(step)
        score_map_z = self._score_map_overlay_z(step, wall_meshes)
        if self._show_score_map:
            self._add_score_map_overlay(score_map, score_map_z)

        # Coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self._origin_frame_size(wall_cfg)
        )
        frame.compute_vertex_normals()
        mat_frame = rendering.MaterialRecord()
        mat_frame.shader = "defaultLit"
        mat_frame.base_color = [1.0, 1.0, 1.0, 1.0]
        self._add_mesh("coord_frame", frame, mat_frame)

        for name, mesh, mat in self._excavator_mesh_entries():
            self._add_mesh(name, mesh, mat)
        for name, frame_mesh, mat in self._excavator_frame_entries():
            self._add_mesh(name, frame_mesh, mat)

        # Target wall (transparent)
        for i, (v, t) in enumerate(wall_meshes):
            wall_mesh = _make_mesh(v, t)
            wall_mesh.translate(self._target_structure_offset)
            self._add_mesh(
                f"wall_{i}",
                wall_mesh,
                _mat(_COLOR_WALL, transparent=True),
            )

        # Already-placed scene stones
        for stone_id in step["scene"]["stone_seq"]:
            sid = int(stone_id)
            mesh_data = self._stone_mesh(sid)
            if mesh_data is None:
                continue
            pose = self._display_pose(step["scene"]["stone_poses"][sid])
            if not np.all(np.isfinite(pose)):
                continue
            mesh = _make_mesh(*mesh_data)
            mesh.transform(pose_to_transformation_matrix(pose))
            mesh.paint_uniform_color(_COLOR_SCENE_STONE[:3])
            self._add_mesh(f"scene_{sid}", mesh, _mat(_COLOR_SCENE_STONE))

        # Candidates: either all score-colored candidates, or only the selected one.
        cands = step["candidates"]
        finite_scores = [c["score"] for c in cands if np.isfinite(c["score"])]
        lo = float(min(finite_scores)) if finite_scores else 0.0
        hi = float(max(finite_scores)) if finite_scores else 1.0

        render_mode = getattr(
            self,
            "_candidate_render_mode",
            "all" if self._render_all_candidates else "selected",
        )
        render_all_candidates = render_mode not in {"selected", "no_select"}
        render_all_opaque = render_mode == "all_opaque"
        render_selected_overlay = render_mode in {"all", "selected"}
        render_candidate_details = render_mode != "no_select"

        if render_all_candidates:
            for i, c in enumerate(cands):
                if i == self._cand_idx and render_selected_overlay:
                    continue  # draw selected last so it's always on top
                sid  = int(c["stone_id"])
                pose_key = (
                    "pose"
                    if self._candidate_pose_key
                    in {"trajectory", "best_sequence", "init_and_solved"}
                    else self._candidate_pose_key
                )
                pose = self._display_pose(c.get(pose_key, c["pose"]))
                mesh_data = self._stone_mesh(sid)
                if mesh_data is None or not np.all(np.isfinite(pose)):
                    continue
                if self._candidate_pose_key == "init_and_solved":
                    self._add_initial_solved_candidate_poses(f"cand_{i}", mesh_data, c)
                    continue
                color = _score_color(c["score"], lo, hi).tolist()
                if render_all_opaque:
                    color[3] = 1.0
                mesh  = _make_mesh(*mesh_data)
                mesh.transform(pose_to_transformation_matrix(pose))
                mesh.paint_uniform_color(color[:3])
                self._add_mesh(
                    f"cand_{i}",
                    mesh,
                    _mat(color, transparent=not render_all_opaque),
                )

        info_text = "—"
        floor_info_lines = []
        floor_data = step.get("floor_fill")
        if isinstance(floor_data, dict):
            floor_info_lines = [
                "Active floor occupancy: "
                f"{float(floor_data.get('occupancy', 0.0)):.3f} / "
                f"{float(floor_data.get('required', 0.0)):.3f}",
                "Active floor z: "
                f"{float(floor_data.get('support_z', 0.0)):.3f}"
                " (inferred)",
            ]

        if self._cand_idx < len(cands):
            c    = cands[self._cand_idx]
            sid  = int(c["stone_id"])
            mesh_data = self._stone_mesh(sid)
            if (
                render_candidate_details
                and mesh_data is not None
                and self._candidate_pose_key == "trajectory"
            ):
                self._add_candidate_trajectory(sc, mesh_data, c)
            if render_candidate_details and self._candidate_pose_key == "best_sequence":
                self._add_candidate_best_sequence(c)

            pose_key = (
                "pose"
                if self._candidate_pose_key
                in {"trajectory", "best_sequence", "init_and_solved"}
                else self._candidate_pose_key
            )
            pose = self._display_pose(c.get(pose_key, c["pose"]))
            if (
                render_candidate_details
                and render_selected_overlay
                and mesh_data is not None
                and np.all(np.isfinite(pose))
            ):
                if self._candidate_pose_key == "init_and_solved":
                    self._add_initial_solved_candidate_poses(
                        "cand_selected", mesh_data, c
                    )
                else:
                    mesh = _make_mesh(*mesh_data)
                    mesh.transform(pose_to_transformation_matrix(pose))
                    mesh.paint_uniform_color(_COLOR_CAND_SELECTED[:3])
                    self._add_mesh("cand_selected", mesh, _mat(_COLOR_CAND_SELECTED))
            if render_candidate_details:
                self._add_scene_motion_trajectory(c)
            if render_candidate_details and self._show_contacts:
                self._add_candidate_contacts(c)
            if render_candidate_details and self._show_pose_solve_contacts:
                self._add_pose_solve_contacts(c)
            if render_candidate_details and self._show_failed_grasps:
                self._add_candidate_failed_grasps(c)

            lines = [
                f"Rank {c['rank']}  |  Score: {c['score']:+.4f}",
                f"Stone ID: {c['stone_id']}",
                f"Status: {self._candidate_status(c)}",
            ]
            if c.get("reward") is not None:
                lines.append(f"Reward:  {c['reward']:.4f}")
            if c.get("visits") is not None:
                lines.append(f"Visits:  {c['visits']}")
            if self._candidate_pose_key == "final_pose":
                lines.append(
                    "Final pose: settled"
                    if c.get("final_pose") is not None
                    else "Final pose: n/a (showing solved)"
                )
            if self._candidate_pose_key == "trajectory":
                lines.append(f"Trajectory: {len(c.get('trajectory', []))} poses")
            if self._candidate_pose_key == "best_sequence":
                lines.append(f"Best sequence: {len(c.get('best_sequence', []))} actions")
            scene_motion = c.get("scene_motion") or {}
            if scene_motion.get("trajectory"):
                lines.append(
                    "Scene motion: stone "
                    f"{int(scene_motion.get('stone_id', -1))}, "
                    f"{len(scene_motion['trajectory'])} poses"
                )
            lines.extend(self._candidate_velocity_info_lines(c))
            if self._sceneid_ground_height is not None:
                lines.append(f"SceneID ground z: {self._sceneid_ground_height:.4f}")
            lines.extend(floor_info_lines)
            lines.extend(self._score_map_info_lines(score_map, score_map_z))
            lines.extend(self._candidate_reward_info_lines(c))
            if self._show_contacts:
                lines.append(f"Contacts: {len(c.get('contact_points', []))}")
            if self._show_pose_solve_contacts:
                pose_contacts = self._pose_solve_contacts(c)
                lines.append(f"Pose solve contacts: {len(pose_contacts)}")
                max_force = self._pose_solve_force_max(pose_contacts)
                if max_force > 0.0:
                    lines.append(f"Pose solve max force: {max_force:.3f}")
            if self._show_failed_grasps:
                n_failed = len(c.get("failed_grasps", []))
                if self._failed_grasp_index is not None and self._failed_grasp_index >= 0:
                    shown = (
                        f"#{self._failed_grasp_index}"
                        if self._failed_grasp_index < n_failed
                        else "none (idx out of range)"
                    )
                    lines.append(f"Failed grasps: {n_failed} (showing {shown})")
                else:
                    lines.append(f"Failed grasps: {n_failed} (showing all)")
            info_text = "\n".join(lines)
        elif bool(step.get("state_only", False)):
            lines = [
                "State checkpoint",
                f"Placed stones: {len(step.get('scene', {}).get('stone_seq', []))}",
            ]
            lines.extend(self._score_map_info_lines(score_map, score_map_z))
            info_text = "\n".join(lines)
        else:
            lines = []
            if self._sceneid_ground_height is not None:
                lines.append(f"SceneID ground z: {self._sceneid_ground_height:.4f}")
            lines.extend(floor_info_lines)
            lines.extend(self._score_map_info_lines(score_map, score_map_z))
            info_text = "\n".join(lines)

        self._info_label.text = info_text

        # Camera — only fit when the step changes, not on every candidate switch
        if self._fit_camera:
            bounds = sc.bounding_box
            if not bounds.is_empty():
                self._sw.setup_camera(60.0, bounds, bounds.get_center())
            else:
                self._sw.look_at([0, 0, 1], [4, 4, 3], [0, 0, 1])
            self._fit_camera = False

    def _candidate_status(self, c: dict) -> str:
        if c.get("motion_failed", False):
            return "Motion planning failed"
        info = c.get("info", {}) or {}
        if info.get("cem_final_sim_failed", False):
            reason = info.get("cem_final_sim_failure", "unknown")
            return f"CEM final sim failed: {reason}"
        if info.get("final_validation_failed", False):
            reason = info.get("final_validation_failure", "unknown")
            return f"Final validation failed: {reason}"
        if not self._candidate_validated(c):
            return "Unvalidated"
        return "Failed" if c.get("failed", False) else "Feasible"

    def _score_map_for_step(self, step: dict) -> dict | None:
        score_map = step.get("score_map")
        if isinstance(score_map, dict) and score_map:
            return score_map
        return self._recompute_score_map_for_step(step)

    def _score_map_environment(self):
        if self._score_map_env is not None:
            return self._score_map_env
        config_path = self._data_path.parent / "config.yml"
        if not config_path.exists():
            return None
        try:
            from agent.env import StoneStackingEnv

            cfg = OmegaConf.load(config_path)
            self._score_map_env = StoneStackingEnv(
                {"cfg": cfg.environment, "n_threads": 1, "build_action_builder": False}
            )
        except Exception as exc:
            print(f"[WARN] Could not build score-map environment: {exc}")
            self._score_map_env = None
        return self._score_map_env

    def _recompute_score_map_for_step(self, step: dict) -> dict | None:
        state = step.get("raw_state")
        if state is None:
            return None
        try:
            step_key = int(step.get("step", self._step_idx))
            attempt_key = step.get("attempt", None)
            cache_key = (step_key, attempt_key, "state")
            if cache_key in self._score_map_cache:
                return self._score_map_cache[cache_key]

            env = self._score_map_environment()
            if env is None:
                return None
            from agent.env.components.action.floor_fill import score_xy_debug_map

            active_floor = self._active_floor_from_step(env, state, step)
            score_map = score_xy_debug_map(
                env.inventory,
                state,
                stone_idx=None,
                active_floor=active_floor,
            )
            score_map["selected_xy"] = np.empty((0, 2), dtype=float)
            score_map["stone_idx"] = None
            score_map["stone_id"] = None
            score_map["n_candidates"] = int(
                len(score_map.get("candidates", []) or [])
            )
            score_map["scope"] = "state"
            score_map = compact_score_map_debug(score_map)
            self._score_map_cache[cache_key] = score_map
            return score_map
        except Exception as exc:
            print(f"[WARN] Could not recompute score map: {exc}")
            return None

    @staticmethod
    def _active_floor_from_step(env, state, step: dict) -> dict | None:
        """Rebuild the active-floor occupancy from the step's stored `floor_fill`.

        The recompute env is built from the session `config.yml`, whose inventory
        only holds the remaining pickable stones. For resumed/SceneID scenes the
        already-placed stones are not in that inventory, so a fresh
        `active_floor_context` resolves zero placed stones and the score map
        collapses to a flat baseline. When the debug step already recorded the
        planning-time occupancy grid, reuse it so the occupancy-driven score
        terms match what MCTS saw. (Heights still come from the config inventory,
        so the height term remains approximate.)
        """
        floor_fill = step.get("floor_fill")
        if not isinstance(floor_fill, dict):
            return None
        occupied = np.asarray(floor_fill.get("occupied", []), dtype=bool)
        if occupied.ndim != 2 or not occupied.any():
            return None
        from agent.env.components.action.floor_fill import (
            neighbor_mask,
            occupancy_grid,
        )

        grid = occupancy_grid(env.inventory, occupied.shape[0])
        if grid is None:
            return None
        lower, upper, xx, yy = grid
        if xx.shape != occupied.shape:
            return None
        return {
            "support_z": float(floor_fill.get("support_z", 0.0)),
            "bottom_tol": float(floor_fill.get("bottom_tol", 0.0)),
            "lower": lower,
            "upper": upper,
            "xx": xx,
            "yy": yy,
            "occupied": occupied,
            "occupied_neighbor": neighbor_mask(occupied),
            "occupancy": float(np.mean(occupied)),
        }

    def _add_score_map_overlay(
        self,
        score_map: dict | None,
        z: float,
    ) -> None:
        if not isinstance(score_map, dict):
            return
        scores = np.asarray(score_map.get("scores", []), dtype=float)
        xs = np.asarray(score_map.get("x_coords", []), dtype=float).reshape(-1)
        ys = np.asarray(score_map.get("y_coords", []), dtype=float).reshape(-1)
        if scores.ndim != 2 or len(xs) < 2 or len(ys) < 2:
            return
        if scores.shape != (len(ys), len(xs)):
            return
        finite = np.isfinite(scores)
        if not np.any(finite):
            return
        valid = np.asarray(score_map.get("valid", np.ones_like(scores)), dtype=bool)
        if valid.shape != scores.shape:
            valid = np.ones_like(scores, dtype=bool)
        render_mask = np.ones_like(scores, dtype=bool)

        x_edges, y_edges = self._score_map_render_edges(xs, ys)
        dx = float(np.median(np.diff(x_edges)))
        dy = float(np.median(np.diff(y_edges)))
        if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
            return
        lo = float(np.nanmin(scores[finite]))
        hi = float(np.nanmax(scores[finite]))
        for iy, ix in np.argwhere(render_mask):
            x0 = float(x_edges[ix])
            x1 = float(x_edges[ix + 1])
            y0 = float(y_edges[iy])
            y1 = float(y_edges[iy + 1])
            if finite[iy, ix]:
                color = _score_color(float(scores[iy, ix]), lo, hi).tolist()
                color[3] = max(float(color[3]), 0.55)
            elif valid[iy, ix]:
                color = list(_COLOR_SCORE_VALID_CELL)
            else:
                color = [0.55, 0.55, 0.55, 0.10]
            mesh = self._floor_cell_mesh(x0, x1, y0, y1, z)
            self._add_mesh(
                f"score_map_{iy}_{ix}",
                mesh,
                _mat(color, transparent=True),
            )

        selected_xy = np.asarray(score_map.get("selected_xy", []), dtype=float)
        if selected_xy.ndim != 2 or selected_xy.shape[1] < 2:
            return
        for i, xy in enumerate(selected_xy[:, :2]):
            if not np.all(np.isfinite(xy)):
                continue
            marker_xy = xy + self._target_structure_offset[:2]
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=max(min(abs(dx), abs(dy)) * 0.25, 0.015)
            )
            sphere.translate(
                [
                    float(marker_xy[0]),
                    float(marker_xy[1]),
                    z + 0.018,
                ]
            )
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(_COLOR_SCORE_SELECTED_XY[:3])
            self._add_mesh(
                f"score_map_selected_xy_{i}",
                sphere,
                _mat(_COLOR_SCORE_SELECTED_XY),
            )

    @staticmethod
    def _grid_edges(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float).reshape(-1)
        if len(coords) < 2:
            step = 1.0
            return np.array([coords[0] - 0.5 * step, coords[0] + 0.5 * step])
        mids = 0.5 * (coords[:-1] + coords[1:])
        first = coords[0] - (mids[0] - coords[0])
        last = coords[-1] + (coords[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]])

    def _score_map_render_edges(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset = self._target_structure_offset
        return self._grid_edges(xs + float(offset[0])), self._grid_edges(
            ys + float(offset[1])
        )

    def _score_map_overlay_z(self, step: dict, wall_meshes: list) -> float:
        offset = self._target_structure_offset
        max_z = float(offset[2])
        if self._sceneid_ground_height is not None:
            max_z = max(max_z, float(self._sceneid_ground_height))
        floor_data = step.get("floor_fill")
        if isinstance(floor_data, dict):
            max_z = max(
                max_z,
                float(floor_data.get("support_z", 0.0)) + float(offset[2]),
            )

        for vertices, _triangles in wall_meshes:
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] < 3 or vertices.size == 0:
                continue
            finite_z = vertices[:, 2][np.isfinite(vertices[:, 2])]
            if len(finite_z) > 0:
                max_z = max(max_z, float(np.max(finite_z)) + float(offset[2]))

        scene = step.get("scene", {}) or {}
        scene_poses = scene.get("stone_poses", {}) or {}
        for stone_id in scene.get("stone_seq", []) or []:
            sid = int(stone_id)
            mesh_data = self._stone_mesh(sid)
            pose = self._display_pose(scene_poses.get(sid, []))
            max_z = max(max_z, self._posed_mesh_max_z(mesh_data, pose))

        return float(max_z + _SCORE_MAP_Z_CLEARANCE)

    @staticmethod
    def _posed_mesh_max_z(mesh_data, pose) -> float:
        if mesh_data is None:
            return -np.inf
        vertices, _triangles = mesh_data
        vertices = np.asarray(vertices, dtype=float)
        pose = np.asarray(pose, dtype=float)
        if (
            vertices.ndim != 2
            or vertices.shape[1] < 3
            or vertices.size == 0
            or pose.shape[0] < 7
            or not np.all(np.isfinite(pose[:7]))
        ):
            return -np.inf
        try:
            transform = pose_to_transformation_matrix(pose[:7])
            homo = np.column_stack([vertices[:, :3], np.ones(len(vertices))])
            z = (homo @ transform.T)[:, 2]
            z = z[np.isfinite(z)]
            return -np.inf if len(z) == 0 else float(np.max(z))
        except Exception:
            return -np.inf

    @staticmethod
    def _score_map_info_lines(score_map: dict | None, z: float) -> list[str]:
        if not isinstance(score_map, dict):
            return ["Score map: unavailable"]
        scores = np.asarray(score_map.get("scores", []), dtype=float)
        finite = np.isfinite(scores)
        if scores.ndim != 2 or not np.any(finite):
            return ["Score map: no finite cells"]
        stone_id = score_map.get("stone_id", None)
        if score_map.get("scope") == "state":
            stone_label = "state"
        elif stone_id is None:
            stone_label = "all stones"
        else:
            stone_label = f"stone {int(stone_id)}"
        return [
            "Score map: "
            f"{stone_label}, {int(np.count_nonzero(finite))} cells, "
            f"range {float(np.nanmin(scores[finite])):.3f}.."
            f"{float(np.nanmax(scores[finite])):.3f}",
            "Score color: blue low, red high",
            "Score map XY: planner coordinates",
            f"Score map z: {float(z):.3f}",
        ]

    @staticmethod
    def _candidate_velocity_info_lines(c: dict) -> list[str]:
        raw = c.get("velocity_integrals", {}) or {}
        values = []
        for stone_id, value in raw.items():
            try:
                value = float(value)
                stone_id = int(stone_id)
            except (TypeError, ValueError):
                continue
            if not np.isnan(value):
                values.append((stone_id, value))
        if not values:
            return []

        target_id = int(c.get("stone_id", -1))
        target_value = next(
            (value for stone_id, value in values if stone_id == target_id),
            None,
        )
        max_stone_id, max_value = max(values, key=lambda item: item[1])
        lines = []
        if target_value is not None:
            lines.append(f"Velocity integral (target): {target_value:.4f}")
        lines.append(
            f"Velocity integral (max): {max_value:.4f} (stone {max_stone_id})"
        )
        return lines

    @staticmethod
    def _candidate_reward_info_lines(c: dict) -> list[str]:
        info = c.get("info", {}) or {}
        keys = (
            "target_IoU",
            "target_IoU_increment",
            "stone_IoU",
            "support_count",
            "support_has_ground",
            "place_robustness_displacement",
            "place_scene_min_gap",
            "place_scene_min_gap_threshold",
            "place_scene_gap_source",
            "place_plane_min_gap",
            "place_plane_min_gap_threshold",
            "place_plane_gap_source",
            "debug_extra_failure",
            "cem_final_sim_failed",
            "cem_final_sim_failure",
            "cem_final_sim_score",
            "cem_final_sim_reward",
            "final_validation_rank",
            "final_validation_prior_score",
            "final_validation_failure",
            "final_validation_scene_motion_retry",
            "final_validation_initial_max_velocity_integral",
            "final_validation_initial_max_velocity_stone_id",
            "final_validation_initial_target_velocity_integral",
            "final_validation_velocity_integral_threshold",
            "simulation_failure_reason",
            "simulation_max_velocity_integral",
            "simulation_max_velocity_stone_id",
            "simulation_target_velocity_integral",
            "simulation_target_settle_position_delta",
            "simulation_target_settle_path_length",
            "simulation_settled",
            "simulation_type",
            "simulation_velocity_integral_threshold",
            "long_simulation_velocity_integral_threshold",
            "final_validation_scene_motion_retry_passed",
            "final_validation_scene_motion_retry_failure",
            "inward_orientation",
            "stability",
        )
        lines = []
        for key in keys:
            if key not in info:
                continue
            value = info[key]
            if isinstance(value, (bool, np.bool_)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, (float, int, np.floating, np.integer)):
                lines.append(f"{key}: {float(value):.4f}")
            else:
                lines.append(f"{key}: {value}")
        return lines

    def _add_candidate_contacts(self, candidate: dict) -> None:
        contacts = candidate.get("contact_points", []) or []
        for i, contact in enumerate(contacts):
            point = self._contact_point(contact)
            if point is None:
                continue
            point = self._display_points(point)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
            sphere.translate(point)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(_COLOR_CONTACT[:3])
            self._add_mesh(f"contact_{i}", sphere, _mat(_COLOR_CONTACT))

            normal = np.asarray(contact.get("normal", []), dtype=float)
            if normal.shape[0] >= 3 and np.all(np.isfinite(normal[:3])):
                norm = float(np.linalg.norm(normal[:3]))
                if norm > 1e-8:
                    end = point + 0.12 * normal[:3] / norm
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(
                        np.vstack([point, end]).astype(float)
                    )
                    line_set.lines = o3d.utility.Vector2iVector(
                        np.asarray([[0, 1]], dtype=np.int32)
                    )
                    self._add_geometry(
                        f"contact_normal_{i}",
                        line_set,
                        _line_mat(_COLOR_CONTACT_NORMAL, width=3.0),
                    )

    def _add_pose_solve_contacts(self, candidate: dict) -> None:
        contacts = self._pose_solve_contacts(candidate)
        if not contacts:
            return
        max_force = self._pose_solve_force_max(contacts)
        force_scale = 0.20 / max(max_force, 1e-9)

        for i, contact in enumerate(contacts):
            point = self._pose_solve_contact_point(contact)
            if point is None:
                continue
            point = self._display_points(point)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.022)
            sphere.translate(point)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(_COLOR_POSE_SOLVE_CONTACT[:3])
            self._add_mesh(
                f"pose_solve_contact_{i}",
                sphere,
                _mat(_COLOR_POSE_SOLVE_CONTACT),
            )

            force = self._pose_solve_contact_force(contact)
            if force is None or max_force <= 0.0:
                continue
            end = point + force_scale * force
            if not np.all(np.isfinite(end)) or np.linalg.norm(end - point) <= 1e-8:
                continue
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(
                np.vstack([point, end]).astype(float)
            )
            line_set.lines = o3d.utility.Vector2iVector(
                np.asarray([[0, 1]], dtype=np.int32)
            )
            self._add_geometry(
                f"pose_solve_force_{i}",
                line_set,
                _line_mat(_COLOR_POSE_SOLVE_FORCE, width=4.0),
            )

    @staticmethod
    def _pose_solve_contacts(candidate: dict) -> list:
        contacts = candidate.get("pose_solve_contacts", []) or []
        return contacts if isinstance(contacts, list) else []

    @classmethod
    def _pose_solve_force_max(cls, contacts: list) -> float:
        norms = []
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            force = cls._pose_solve_contact_force(contact)
            if force is not None:
                norms.append(float(np.linalg.norm(force)))
        return float(max(norms)) if norms else 0.0

    @staticmethod
    def _pose_solve_contact_point(contact: dict):
        if not isinstance(contact, dict):
            return None
        for key in ("point", "contact_point"):
            value = np.asarray(contact.get(key, []), dtype=float)
            if value.shape[0] >= 3 and np.all(np.isfinite(value[:3])):
                return value[:3].copy()
        return CandidateViewer._contact_point(contact)

    @staticmethod
    def _pose_solve_contact_force(contact: dict):
        if not isinstance(contact, dict):
            return None
        value = np.asarray(contact.get("force", []), dtype=float)
        if value.shape[0] >= 3 and np.all(np.isfinite(value[:3])):
            return value[:3].copy()
        return None

    def _add_candidate_failed_grasps(self, candidate: dict) -> None:
        grasps = candidate.get("failed_grasps", []) or []
        if not grasps:
            return
        idx = self._failed_grasp_index
        if idx is not None and idx >= 0:
            if idx >= len(grasps):
                return
            indexed = [(idx, grasps[idx])]
        else:
            indexed = list(enumerate(grasps))
        self._load_gripper_model()
        if self._gripper_model is None:
            return
        from model import update_urdf_mesh

        for i, grasp in indexed:
            pose = np.asarray(grasp.get("pose", []), dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                continue
            try:
                opening_angle = float(grasp.get("opening_angle", 0.0))
            except (TypeError, ValueError):
                opening_angle = 0.0
            q_gripper = np.full(2, opening_angle, dtype=float)
            try:
                meshes = update_urdf_mesh(
                    self._gripper_model, self._gripper_meshes, q_gripper
                )
            except Exception:
                continue
            for link_name, mesh in meshes.items():
                mesh.transform(pose)
                mesh.paint_uniform_color(_COLOR_FAILED_GRASP[:3])
                self._add_mesh(
                    f"failed_grasp_{i}_{link_name}",
                    mesh,
                    _mat(_COLOR_FAILED_GRASP, transparent=True),
                )

    @staticmethod
    def _contact_point(contact: dict):
        pts = []
        for key in ("s_1", "s_2"):
            value = np.asarray(contact.get(key, []), dtype=float)
            if value.shape[0] >= 3 and np.all(np.isfinite(value[:3])):
                pts.append(value[:3])
        if not pts:
            return None
        return np.mean(np.asarray(pts, dtype=float), axis=0)

    def _add_candidate_best_sequence(self, candidate: dict) -> None:
        sequence = candidate.get("best_sequence", []) or []
        if not sequence:
            return
        for i, item in enumerate(sequence):
            sid = int(item.get("stone_id", -1))
            mesh_data = self._stone_mesh(sid)
            pose = self._display_pose(item.get("pose", []))
            if mesh_data is None or pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
                continue

            if i == 0:
                color = _COLOR_CAND_SELECTED
                name = "best_seq_selected"
            else:
                color = list(_COLOR_SEQUENCE_FUTURE)
                color[3] = min(0.25 + 0.08 * i, 0.70)
                name = f"best_seq_{i}"
            mesh = _make_mesh(*mesh_data)
            mesh.transform(pose_to_transformation_matrix(pose[:7]))
            mesh.paint_uniform_color(color[:3])
            self._add_mesh(name, mesh, _mat(color, transparent=color[3] < 1.0))

        points = [
            np.asarray(item.get("pose", []), dtype=float)[:3]
            for item in sequence
            if np.asarray(item.get("pose", []), dtype=float).shape[0] >= 7
            and np.all(np.isfinite(np.asarray(item.get("pose", []), dtype=float)[:7]))
        ]
        if len(points) >= 2:
            self._add_sequence_path(self._display_points(np.asarray(points, dtype=float)))

    def _add_sequence_path(self, points: np.ndarray) -> None:
        lines = np.column_stack(
            [np.arange(len(points) - 1), np.arange(1, len(points))]
        )
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points.astype(float))
        line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
        self._add_geometry(
            "best_seq_path",
            line_set,
            _line_mat(_COLOR_SEQUENCE_FUTURE, width=4.0),
        )

    def _add_initial_solved_candidate_poses(
        self,
        prefix: str,
        mesh_data,
        candidate: dict,
    ) -> None:
        init_pose = candidate.get("init_pose")
        if init_pose is not None:
            self._add_candidate_pose_mesh(
                f"{prefix}_initial",
                mesh_data,
                init_pose,
                _COLOR_CAND_INITIAL,
            )
        self._add_candidate_pose_mesh(
            f"{prefix}_solved",
            mesh_data,
            candidate.get("solved_pose", candidate.get("pose", [])),
            _COLOR_CAND_SOLVED,
        )

    def _add_candidate_pose_mesh(
        self,
        name: str,
        mesh_data,
        pose,
        color: list,
        transparent: bool = False,
    ) -> None:
        pose = self._display_pose(pose)
        if (
            pose.ndim != 1
            or pose.shape[0] < 7
            or not np.all(np.isfinite(pose[:7]))
        ):
            return
        mesh = _make_mesh(*mesh_data)
        mesh.transform(pose_to_transformation_matrix(pose[:7]))
        mesh.paint_uniform_color(color[:3])
        self._add_mesh(name, mesh, _mat(color, transparent=transparent))

    def _add_candidate_trajectory(self, sc, mesh_data, candidate: dict) -> None:
        poses = candidate.get("trajectory", [])
        if not poses:
            return
        pose_array = np.asarray(poses, dtype=float)
        valid = (
            pose_array.ndim == 2
            and pose_array.shape[1] >= 7
            and np.all(np.isfinite(pose_array[:, :7]), axis=1)
        )
        if isinstance(valid, np.ndarray):
            pose_array = pose_array[valid]
        if len(pose_array) == 0:
            return

        self._add_trajectory_path(self._display_points(pose_array[:, :3]))

        max_meshes = 24
        if len(pose_array) <= max_meshes:
            indices = list(range(len(pose_array)))
        else:
            indices = np.linspace(0, len(pose_array) - 1, max_meshes, dtype=int).tolist()

        for j, pose_idx in enumerate(indices):
            pose = self._display_pose(pose_array[pose_idx])
            if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
                continue
            alpha = 0.08 + 0.22 * ((j + 1) / max(len(indices), 1))
            color = list(_COLOR_TRAJECTORY)
            color[3] = alpha
            mesh = _make_mesh(*mesh_data)
            mesh.transform(pose_to_transformation_matrix(pose[:7]))
            mesh.paint_uniform_color(color[:3])
            self._add_mesh(
                f"cand_traj_{j}",
                mesh,
                _mat(color, transparent=True),
            )

    def _add_scene_motion_trajectory(self, candidate: dict) -> None:
        scene_motion = candidate.get("scene_motion") or {}
        poses = np.asarray(scene_motion.get("trajectory", []), dtype=float)
        if poses.ndim != 2 or poses.shape[1] < 7:
            return
        poses = poses[np.all(np.isfinite(poses[:, :7]), axis=1), :7]
        if len(poses) == 0:
            return

        stone_id = int(scene_motion.get("stone_id", -1))
        mesh_data = self._stone_mesh(stone_id)
        if mesh_data is not None:
            final_pose = self._display_pose(poses[-1])
            mesh = _make_mesh(*mesh_data)
            mesh.transform(pose_to_transformation_matrix(final_pose))
            mesh.paint_uniform_color(_COLOR_SCENE_MOTION[:3])
            self._add_mesh(
                "scene_motion_final",
                mesh,
                _mat(_COLOR_SCENE_MOTION, transparent=True),
            )

        points = self._display_points(poses[:, :3])
        if len(points) > 256:
            points = points[np.linspace(0, len(points) - 1, 256, dtype=int)]
        if len(points) < 2:
            return
        lines = np.column_stack(
            [np.arange(len(points) - 1), np.arange(1, len(points))]
        )
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points.astype(float))
        line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
        self._add_geometry(
            "scene_motion_path",
            line_set,
            _line_mat(_COLOR_SCENE_MOTION_LINE, width=6.0),
        )

    def _add_trajectory_path(self, points: np.ndarray) -> None:
        if len(points) < 2:
            return
        max_points = 256
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = points[indices]
        lines = np.column_stack(
            [np.arange(len(points) - 1), np.arange(1, len(points))]
        )
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points.astype(float))
        line_set.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
        self._add_geometry("cand_traj_path", line_set, _line_mat(_COLOR_TRAJ_LINE))

        for name, point, color in (
            ("cand_traj_start", points[0], _COLOR_TRAJ_START),
            ("cand_traj_end", points[-1], _COLOR_TRAJ_END),
        ):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
            sphere.translate(point)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color[:3])
            self._add_mesh(name, sphere, _mat(color))


# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for MCTS debug state"
    )
    parser.add_argument(
        "data",
        help=(
            "path to a debug_state.pkl or a State checkpoint inside an "
            "ablation case"
        ),
    )
    parser.add_argument(
        "--mesh-source",
        choices=("auto", "dsf", "ply"),
        default="auto",
        help="mesh source to display; requires debug pkl saved with PLY meshes for ply",
    )
    parser.add_argument(
        "--excavator",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Show excavator joint controls. auto enables them for generated "
            "sequence session pickles with planning_params.pkl."
        ),
    )
    args = parser.parse_args()
    try:
        CandidateViewer(
            args.data,
            mesh_source=args.mesh_source,
            excavator=args.excavator,
        )
    except (EOFError, pickle.UnpicklingError, TypeError, ValueError) as exc:
        path = Path(args.data)
        size = path.stat().st_size if path.exists() else 0
        parser.exit(
            2,
            "error: could not load debug pickle "
            f"{path} ({size} bytes): {type(exc).__name__}: {exc}\n"
            "Use a candidate debug pickle, or a State checkpoint under an "
            "ablation case that still contains its result JSON.\n",
        )


if __name__ == "__main__":
    main()
