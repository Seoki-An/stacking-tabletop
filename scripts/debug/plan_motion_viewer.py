#!/usr/bin/env python3
"""Interactive replay viewer for saved plan motions.

Usage:
    direnv exec . python -m scripts.debug.plan_motion_viewer 260701_5
    direnv exec . python -m scripts.debug.plan_motion_viewer sessions/260701/plan_260701_5
    direnv exec . python -m scripts.debug.plan_motion_viewer 260701_5 --video step
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

os.environ.setdefault("STACKING_LIVE_JOINT_VIEWER", "0")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from planning.execution import PlanningWorker
from planning.execution_common import (  # noqa: E402
    C_GHOST,
    C_GRIPPER,
    C_GROUND,
    C_REPLANNED_TRAJECTORY,
    C_STONE,
    C_TARGET,
    PLANNING_PREVIEW_MAX_MARKERS,
)

_PANEL_W = 330
_SPLITTER_W = 10
_TARGET_NAME = "replay_target"
_DEFAULT_FPS = 30.0
_DEFAULT_VIDEO_SUBDIR = os.path.join("videos", "replay")


@dataclass
class ReplayStep:
    index: int
    target_id: int
    q_path: list[np.ndarray]
    target_path: list[np.ndarray]
    geometries: list
    title: str
    summary: str


def _mat(color, alpha: float = 1.0) -> rendering.MaterialRecord:
    color = list(color or [0.7, 0.7, 0.75])[:3]
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mat = rendering.MaterialRecord()
    mat.base_color = color + [alpha]
    if alpha < 1.0:
        mat.shader = "defaultLitTransparency"
        mat.has_alpha = True
    else:
        mat.shader = "defaultLit"
    return mat


def _line_mat(color, width: float = 4.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.base_color = list(color or [1.0, 1.0, 1.0])[:3] + [1.0]
    mat.line_width = width
    return mat


def _split_entry(entry):
    geom = entry
    color = None
    meta = {}
    style = None
    if isinstance(entry, tuple):
        geom = entry[0]
        if len(entry) > 1:
            color = list(entry[1])[:3]
        for extra in entry[2:]:
            if isinstance(extra, dict):
                meta.update(extra)
            elif isinstance(extra, str):
                style = extra
    if style:
        meta["style"] = style
    return geom, color, meta


def _mesh_wireframe(mesh):
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    lineset = o3d.geometry.LineSet()
    lineset.points = mesh.vertices
    if triangles.size == 0:
        lineset.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
        return lineset
    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lineset.lines = o3d.utility.Vector2iVector(edges)
    return lineset


def _has_renderable_geometry(geom) -> bool:
    try:
        if hasattr(geom, "lines"):
            points = np.asarray(geom.points, dtype=np.float64)
            lines = np.asarray(geom.lines, dtype=np.int64)
            return (
                points.ndim == 2
                and points.shape[1] == 3
                and lines.size > 0
                and np.all(np.isfinite(points))
            )
        if hasattr(geom, "vertices"):
            vertices = np.asarray(geom.vertices, dtype=np.float64)
            triangles = np.asarray(getattr(geom, "triangles", []), dtype=np.int64)
            return (
                vertices.ndim == 2
                and vertices.shape[1] == 3
                and len(vertices) > 0
                and len(triangles) > 0
                and np.all(np.isfinite(vertices))
            )
        if hasattr(geom, "points"):
            points = np.asarray(geom.points, dtype=np.float64)
            return (
                points.ndim == 2
                and points.shape[1] == 3
                and len(points) > 0
                and np.all(np.isfinite(points))
            )
    except Exception:
        return False
    return False


def _camera_array(value, fallback) -> np.ndarray:
    if value is None:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(3)


class PlanReplayWorker(PlanningWorker):
    """PlanningWorker subset that loads saved plans without ROS/control nodes."""

    def _initialize_ros_and_models(self):
        cfg_path = os.path.join(self.plan_dir, "config.yml")
        if not os.path.exists(cfg_path):
            cfg_path = "agent/configs/config.yml"
        self._log(f"Replay config: {cfg_path}")
        cfg = self.OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.asset_dir = cfg.environment.data.load_dir

        old_env_ground_height = self.environment_ground_height(cfg.environment)
        self.set_environment_ground_height(cfg.environment, self.place_plane_height)
        if abs(old_env_ground_height - self.place_plane_height) > 1e-9:
            self._log(
                "Updating replay environment support ground_z: "
                f"{old_env_ground_height:.4f} -> {self.place_plane_height:.4f}"
            )

        from agent.env import StoneStackingEnv

        n_threads = self.resolve_thread_count(20, cfg)
        env = StoneStackingEnv(
            {
                "cfg": cfg.environment,
                "n_threads": n_threads,
                "build_action_builder": False,
            }
        )
        env.reset()
        self.integrated_planner = SimpleNamespace(env=env)
        self.place_body_ids_by_stone: dict[int, int] = {}

        self.context, _ = self.get_planner(
            self.pick_plane_height,
            self.place_plane_height,
            n_threads=n_threads,
        )
        self.posegen = self._create_posegen()
        self.excavator_model, self.excavator_meshes = self.get_excavator_model()
        (
            self.stone_dsf_meshes,
            self.stone_configs,
            self.stone_pcds,
            self.stone_meshes,
        ) = self.get_stone_model(self.asset_dir)
        self.gripper_model, self.gripper_meshes = self.get_gripper_model()

    def setup_for_replay(self):
        self._load_runtime_dependencies()
        self._load_plan_files()
        self._initialize_ros_and_models()
        self._initialize_scene()
        self._field_configs_by_stone = {
            int(stone_id): copy.deepcopy(config)
            for stone_id, config in self.scene_configs.items()
        }

    def build_step(
        self,
        step_index: int,
        show_trajectory: bool = True,
        show_structure: bool = True,
    ) -> ReplayStep:
        if not (0 <= step_index < len(self.action_sequence)):
            raise IndexError(f"step index out of range: {step_index}")

        self._restore_scene_prefix(step_index)
        action = self.action_sequence[step_index]
        target_id = int(action["stone_id"])
        pick_config, place_config = self._build_pick_and_place_configs(
            step_index,
            target_id,
            action,
        )
        result = self._saved_motion_result_for_step(
            step_index,
            pick_config,
            place_config,
        )
        if result is None:
            raise ValueError(f"step {step_index + 1} has no replayable saved motion")

        q_path, target_path = self.generate_path_with_opening_angle(
            result,
            self.n_opening_angle,
        )
        q_path = [self._q_with_opening(q) for q in q_path]
        target_path = [np.asarray(T, dtype=np.float64) for T in target_path]
        n = min(len(q_path), len(target_path))
        q_path = q_path[:n]
        target_path = target_path[:n]
        if n == 0:
            raise ValueError(f"step {step_index + 1} has empty replay paths")

        self._viz_target_id = target_id
        self._viz_target_mode = None
        self._viz_pick_pose_T = pick_config.pose.as_matrix()
        self._viz_place_pose_T = place_config.pose.as_matrix()
        self._viz_opening_angle = float(result.grasp_sequence[0].opening_angle)

        q0 = q_path[0]
        geoms = (
            [(self.ground_mesh, C_GROUND)]
            + self._excavator_at(q0)
            + self._other_stone_entries()
        )
        if show_structure:
            geoms += self._target_structure_entries()
        geoms += [
            self._transform_entry(
                self._copy_base_stone_mesh(target_id),
                C_TARGET,
                _TARGET_NAME,
                target_path[0],
            )
        ]

        if show_trajectory:
            geoms += self._trajectory_entries(target_id, q_path, target_path, result)
        geoms += [self.origin_frame]

        summary = self.motion_result_summary(result)
        title = f"Step {step_index + 1}: stone {target_id}"
        return ReplayStep(
            index=step_index,
            target_id=target_id,
            q_path=q_path,
            target_path=target_path,
            geometries=geoms,
            title=title,
            summary=summary,
        )

    def frame_transforms(self, replay_step: ReplayStep, frame_index: int) -> dict:
        frame_index = int(np.clip(frame_index, 0, len(replay_step.q_path) - 1))
        q = replay_step.q_path[frame_index]
        transforms = {
            self._excavator_live_name(name): transform
            for name, transform in self._excavator_link_transforms(q).items()
        }
        transforms[_TARGET_NAME] = replay_step.target_path[frame_index]
        return transforms

    def save_step_video(
        self,
        step_index: int,
        save_path: str,
        fps: float = _DEFAULT_FPS,
    ) -> None:
        replay_step = self.build_step(
            step_index,
            show_trajectory=False,
            show_structure=False,
        )
        scene_meshes = {
            stone_id: mesh
            for stone_id, mesh in self.scene_meshes.items()
            if int(stone_id) != int(replay_step.target_id)
        }
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        self.trajectory_visualization_with_target(
            replay_step.q_path,
            replay_step.target_path,
            self.excavator_model,
            self.excavator_meshes,
            scene_meshes,
            self._copy_base_stone_mesh(replay_step.target_id),
            save_path,
            self.camera_center,
            self.camera_position,
            wall_meshes=self._wall_display_meshes(),
            fps=float(fps),
        )

    def step_label(self, step_index: int) -> str:
        action = self.action_sequence[step_index]
        stone_id = int(action.get("stone_id", -1))
        return f"{step_index + 1:02d}  stone {stone_id}"

    def _restore_scene_prefix(self, step_index: int) -> None:
        self.scene_meshes = {}
        self.scene_configs = {}
        for stone_id, config in self._field_configs_by_stone.items():
            self._set_scene_stone_pose(stone_id, config)

        for action_i, action in enumerate(self.action_sequence[:step_index]):
            try:
                stone_id = int(action["stone_id"])
            except (TypeError, KeyError, ValueError):
                continue
            if stone_id not in self._field_configs_by_stone:
                continue
            place_pose = self._place_pose_for_action(action, action_i)
            config = copy.deepcopy(self._field_configs_by_stone[stone_id])
            config.pose.setPosition(place_pose[:3])
            config.pose.setOrientation(place_pose[3:7])
            self._set_scene_stone_pose(stone_id, config)

    def _trajectory_entries(self, target_id, q_path, target_path, result):
        target_markers = self._target_trajectory_markers(
            target_path,
            C_REPLANNED_TRAJECTORY,
            max_markers=PLANNING_PREVIEW_MAX_MARKERS,
        )
        lines = [
            line
            for line in (
                self._trajectory_lineset(
                    [self._grip_body_world(q_t)[:3, 3] for q_t in q_path],
                    C_GRIPPER,
                ),
                self._trajectory_lineset(
                    [pose_t[:3, 3] for pose_t in target_path],
                    C_REPLANNED_TRAJECTORY,
                ),
            )
            if line is not None
        ]
        settled_entries = self._settled_release_overlay_entries(
            target_id,
            result,
            max_markers=PLANNING_PREVIEW_MAX_MARKERS,
        )
        grasp_entries = []
        try:
            opening = float(result.grasp_sequence[0].opening_angle)
            q_gripper = np.ones(2) * opening
            meshes = self.update_urdf_mesh(
                self.gripper_model,
                self.gripper_meshes,
                q_gripper,
            )
            grasp_T = result.grasp_sequence[0].pose.as_matrix()
            for mesh in meshes.values():
                mesh.transform(grasp_T)
                grasp_entries.append((mesh, C_GRIPPER))
        except Exception:
            pass
        return grasp_entries + lines + target_markers + settled_entries

    def _wall_display_meshes(self) -> dict[str, o3d.geometry.TriangleMesh]:
        wall_meshes = {}
        offset = np.asarray(self.target_structure_offset[:2], dtype=np.float64)
        for i, geom in enumerate(self.integrated_planner.env.inventory.target_wall.geometries):
            mesh = copy.deepcopy(geom.get_mesh())
            mesh.translate([float(offset[0]), float(offset[1]), 0.0])
            mesh.compute_vertex_normals()
            wall_meshes[f"wall_{i}"] = mesh
        return wall_meshes


class PlanMotionViewer:
    def __init__(
        self,
        worker: PlanReplayWorker,
        initial_step: int,
        fps: float,
        camera_center=None,
        camera_position=None,
    ):
        self.worker = worker
        self._step_index = int(np.clip(initial_step, 0, len(worker.action_sequence) - 1))
        self._frame_index = 0
        self._current: ReplayStep | None = None
        self._show_trajectory = True
        self._show_structure = True
        self._fit_camera = True
        self._playing = False
        self._fps = float(fps)
        self._stop_event = threading.Event()
        self._play_thread = None
        self._updating_controls = False
        self._custom_camera_enabled = (
            camera_center is not None or camera_position is not None
        )
        self._camera_center = _camera_array(camera_center, worker.camera_center)
        self._camera_position = _camera_array(camera_position, worker.camera_position)

        app = gui.Application.instance
        app.initialize()
        self._build_window()
        self._load_step(self._step_index, fit_camera=True)
        self._start_playback_thread()
        app.run()

    def _build_window(self):
        self._win = gui.Application.instance.create_window(
            "Saved Plan Motion Replay",
            1450,
            900,
        )
        self._win.set_on_close(self._on_close)
        em = self._win.theme.font_size
        spacing = int(0.45 * em)
        margin = int(0.55 * em)

        self._sw = gui.SceneWidget()
        self._sw.scene = rendering.Open3DScene(self._win.renderer)
        self._sw.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self._sw.scene.scene.set_indirect_light_intensity(30000)
        self._win.add_child(self._sw)

        self._splitter = gui.Vert()
        self._splitter.background_color = gui.Color(0.22, 0.22, 0.22, 1.0)
        self._win.add_child(self._splitter)

        panel = gui.Vert(spacing, gui.Margins(margin, margin, margin, margin))
        panel.add_child(gui.Label("Plan Motion Replay"))
        self._plan_label = gui.Label(Path(self.worker.plan_dir).name)
        panel.add_child(self._plan_label)
        panel.add_fixed(spacing)

        step_row = gui.Horiz(spacing)
        self._prev_step_btn = gui.Button("Prev")
        self._next_step_btn = gui.Button("Next")
        self._prev_step_btn.set_on_clicked(
            lambda: self._select_step(self._step_index - 1)
        )
        self._next_step_btn.set_on_clicked(
            lambda: self._select_step(self._step_index + 1)
        )
        step_row.add_child(self._prev_step_btn)
        step_row.add_child(self._next_step_btn)
        panel.add_child(step_row)

        self._step_list = gui.ListView()
        self._step_list.set_max_visible_items(14)
        self._step_items = [
            self.worker.step_label(i) for i in range(len(self.worker.action_sequence))
        ]
        self._step_list.set_items(self._step_items)
        self._step_list.set_on_selection_changed(self._on_step_selected)
        panel.add_child(self._step_list)
        panel.add_fixed(spacing)

        frame_row = gui.Horiz(spacing)
        self._play_btn = gui.Button("Play")
        self._play_btn.set_on_clicked(self._toggle_play)
        self._reset_btn = gui.Button("Restart")
        self._reset_btn.set_on_clicked(lambda: self._set_frame(0))
        frame_row.add_child(self._play_btn)
        frame_row.add_child(self._reset_btn)
        panel.add_child(frame_row)

        self._frame_label = gui.Label("Frame: -")
        panel.add_child(self._frame_label)
        self._frame_slider = gui.Slider(gui.Slider.INT)
        self._frame_slider.set_on_value_changed(self._on_frame_slider_changed)
        panel.add_child(self._frame_slider)

        fps_row = gui.Horiz(spacing)
        fps_row.add_child(gui.Label("FPS"))
        self._fps_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._fps_edit.set_limits(1.0, 60.0)
        self._fps_edit.double_value = self._fps
        self._fps_edit.set_on_value_changed(self._on_fps_changed)
        fps_row.add_child(self._fps_edit)
        panel.add_child(fps_row)

        self._loop_cb = gui.Checkbox("Loop step")
        self._loop_cb.checked = True
        panel.add_child(self._loop_cb)

        self._trajectory_cb = gui.Checkbox("Show trajectory")
        self._trajectory_cb.checked = self._show_trajectory
        self._trajectory_cb.set_on_checked(self._on_trajectory_checked)
        panel.add_child(self._trajectory_cb)

        self._structure_cb = gui.Checkbox("Show planned structure")
        self._structure_cb.checked = self._show_structure
        self._structure_cb.set_on_checked(self._on_structure_checked)
        panel.add_child(self._structure_cb)

        self._highpoly_cb = gui.Checkbox("High-poly stones")
        self._highpoly_cb.checked = bool(self.worker._stone_highpoly_mesh_mode)
        self._highpoly_cb.set_on_checked(self._on_highpoly_checked)
        panel.add_child(self._highpoly_cb)

        self._fit_btn = gui.Button("Fit Camera")
        self._fit_btn.set_on_clicked(self._fit_camera_now)
        panel.add_child(self._fit_btn)
        panel.add_fixed(spacing)

        self._info_label = gui.Label("")
        panel.add_child(self._info_label)

        self._panel = panel
        self._win.add_child(panel)
        self._win.set_on_layout(self._on_layout)

    def _on_layout(self, _ctx):
        rect = self._win.content_rect
        panel_w = min(_PANEL_W, max(240, int(rect.width * 0.45)))
        self._panel.frame = gui.Rect(rect.x, rect.y, panel_w, rect.height)
        self._splitter.frame = gui.Rect(
            rect.x + panel_w,
            rect.y,
            _SPLITTER_W,
            rect.height,
        )
        self._sw.frame = gui.Rect(
            rect.x + panel_w + _SPLITTER_W,
            rect.y,
            rect.width - panel_w - _SPLITTER_W,
            rect.height,
        )

    def _on_close(self):
        self._stop_event.set()
        return True

    def _start_playback_thread(self):
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _play_loop(self):
        app = gui.Application.instance
        while not self._stop_event.is_set():
            if not self._playing:
                time.sleep(0.03)
                continue
            app.post_to_main_thread(self._win, self._advance_frame)
            time.sleep(1.0 / max(1.0, self._fps))

    def _toggle_play(self):
        self._playing = not self._playing
        self._play_btn.text = "Pause" if self._playing else "Play"

    def _advance_frame(self):
        if self._current is None:
            return
        next_frame = self._frame_index + 1
        if next_frame >= len(self._current.q_path):
            if self._loop_cb.checked:
                next_frame = 0
            else:
                self._playing = False
                self._play_btn.text = "Play"
                return
        self._set_frame(next_frame)

    def _on_step_selected(self, item: str, _dbl: bool):
        if self._updating_controls:
            return
        try:
            idx = self._step_items.index(item)
        except ValueError:
            return
        self._select_step(idx)

    def _select_step(self, idx: int):
        idx = int(np.clip(idx, 0, len(self.worker.action_sequence) - 1))
        if idx == self._step_index:
            return
        self._step_index = idx
        self._load_step(idx, fit_camera=True)

    def _load_step(self, idx: int, fit_camera: bool):
        self._playing = False
        self._play_btn.text = "Play"
        self._fit_camera = fit_camera
        self._current = self.worker.build_step(
            idx,
            show_trajectory=self._show_trajectory,
            show_structure=self._show_structure,
        )
        self._frame_index = 0
        self._render_layout()
        self._sync_controls()
        self._set_frame(0)

    def _render_layout(self):
        if self._current is None:
            return
        sc = self._sw.scene
        sc.clear_geometry()
        for i, entry in enumerate(self._current.geometries):
            self._add_entry(i, entry)
        if self._fit_camera:
            bounds = sc.bounding_box
            if self._custom_camera_enabled:
                self._sw.look_at(
                    self._camera_center.tolist(),
                    self._camera_position.tolist(),
                    [0.0, 0.0, 1.0],
                )
            elif not bounds.is_empty():
                self._sw.setup_camera(60.0, bounds, bounds.get_center())
            self._fit_camera = False

    def _add_entry(self, index: int, entry):
        geom, color, meta = _split_entry(entry)
        if geom is None:
            return
        geom = copy.deepcopy(geom)
        if meta.get("style") == "wireframe" and hasattr(geom, "vertices"):
            geom = _mesh_wireframe(geom)
        if not _has_renderable_geometry(geom):
            return

        name = str(meta.get("name") or f"g_{index}")
        alpha = float(meta.get("alpha", 1.0))
        if meta.get("style") == "wireframe" or hasattr(geom, "lines"):
            mat = _line_mat(color or C_GHOST)
        elif hasattr(geom, "points") and not hasattr(geom, "triangles"):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = list(color or C_STONE)[:3] + [alpha]
        else:
            if hasattr(geom, "compute_vertex_normals") and not geom.has_vertex_normals():
                geom.compute_vertex_normals()
            mat = _mat(color or C_STONE, alpha=alpha)
        try:
            self._sw.scene.add_geometry(name, geom, mat)
            transform = meta.get("transform")
            if transform is not None:
                self._set_geometry_transform(name, transform)
        except Exception:
            return

    def _set_frame(self, frame_index: int):
        if self._current is None:
            return
        frame_index = int(np.clip(frame_index, 0, len(self._current.q_path) - 1))
        self._frame_index = frame_index
        transforms = self.worker.frame_transforms(self._current, frame_index)
        for name, transform in transforms.items():
            self._set_geometry_transform(name, transform)
        self._sync_frame_controls()
        self._win.post_redraw()

    def _set_geometry_transform(self, name: str, transform):
        try:
            if (
                hasattr(self._sw.scene, "has_geometry")
                and not self._sw.scene.has_geometry(name)
            ):
                return
            transform = np.asarray(transform, dtype=np.float64)
            if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
                self._sw.scene.set_geometry_transform(name, transform)
        except Exception:
            return

    def _sync_controls(self):
        if self._current is None:
            return
        self._updating_controls = True
        try:
            self._step_list.selected_index = self._step_index
            max_frame = max(0, len(self._current.q_path) - 1)
            self._frame_slider.set_limits(0, max_frame)
            self._info_label.text = (
                f"{self._current.title}\n"
                f"Samples: {len(self._current.q_path)}\n"
                f"{self._current.summary}"
            )
            self._prev_step_btn.enabled = self._step_index > 0
            self._next_step_btn.enabled = (
                self._step_index + 1 < len(self.worker.action_sequence)
            )
        finally:
            self._updating_controls = False

    def _sync_frame_controls(self):
        if self._current is None:
            return
        self._updating_controls = True
        try:
            total = len(self._current.q_path)
            self._frame_slider.int_value = self._frame_index
            self._frame_label.text = f"Frame: {self._frame_index + 1} / {total}"
        finally:
            self._updating_controls = False

    def _on_frame_slider_changed(self, value):
        if self._updating_controls:
            return
        self._set_frame(int(value))

    def _on_fps_changed(self, _value):
        self._fps = float(np.clip(self._fps_edit.double_value, 1.0, 60.0))

    def _on_trajectory_checked(self, checked: bool):
        self._show_trajectory = bool(checked)
        self._load_step(self._step_index, fit_camera=False)

    def _on_structure_checked(self, checked: bool):
        self._show_structure = bool(checked)
        self._load_step(self._step_index, fit_camera=False)

    def _on_highpoly_checked(self, checked: bool):
        self.worker._stone_highpoly_mesh_mode = bool(checked)
        self._load_step(self._step_index, fit_camera=False)

    def _fit_camera_now(self):
        self._custom_camera_enabled = False
        self._fit_camera = True
        self._render_layout()
        self._set_frame(self._frame_index)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay saved generate_sequence.py motion results."
    )
    parser.add_argument(
        "plan",
        help=(
            "Plan id or path. Accepted forms include 260701_5, "
            "plan_260701_5, and sessions/260701/plan_260701_5."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="1-based step to show first.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=_DEFAULT_FPS,
        help="Initial playback FPS, and output FPS when --video is used.",
    )
    parser.add_argument(
        "--video",
        choices=["step", "all"],
        default=None,
        help=(
            "Render MP4 video(s) and exit instead of opening the interactive "
            "viewer. 'step' saves --step only; 'all' saves every motion."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Output directory for --video. Defaults to PLAN_DIR/videos/replay.",
    )
    parser.add_argument(
        "--highpoly-stones",
        action="store_true",
        help="Render stones with high-poly PLY meshes when available.",
    )
    parser.add_argument(
        "--camera-center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Camera look-at point for video and initial interactive view.",
    )
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Camera eye position for video and initial interactive view.",
    )
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be >= 1")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be > 0")
    return args


def _video_dir(worker: PlanReplayWorker, requested: str | None) -> str:
    if requested is not None:
        return os.path.abspath(os.path.expanduser(requested))
    return os.path.join(worker.plan_dir, _DEFAULT_VIDEO_SUBDIR)


def _save_videos(
    worker: PlanReplayWorker,
    video_mode: str,
    video_dir: str | None,
    fps: float,
    initial_step: int,
) -> None:
    out_dir = _video_dir(worker, video_dir)
    if video_mode == "step":
        indices = [initial_step]
    else:
        indices = list(range(len(worker.action_sequence)))

    print(f"Saving replay video(s) to {out_dir}")
    for idx in indices:
        save_path = os.path.join(out_dir, f"step_{idx + 1}.mp4")
        print(f"[{idx + 1}/{len(worker.action_sequence)}] {save_path}")
        worker.save_step_video(idx, save_path, fps=fps)


def _video_process_main(
    plan: str,
    step: int,
    video_mode: str,
    video_dir: str | None,
    fps: float,
    highpoly_stones: bool,
    camera_center,
    camera_position,
) -> None:
    worker = PlanReplayWorker(
        plan_id=plan,
        start_step=None,
        on_log=lambda msg: print(msg),
    )
    try:
        worker.setup_for_replay()
        if not worker.action_sequence:
            raise ValueError(f"No actions found in plan: {worker.plan_dir}")
        worker._stone_highpoly_mesh_mode = bool(highpoly_stones)
        worker.camera_center = _camera_array(camera_center, worker.camera_center)
        worker.camera_position = _camera_array(
            camera_position,
            worker.camera_position,
        )
        initial_step = min(step, len(worker.action_sequence)) - 1
        _save_videos(worker, video_mode, video_dir, fps, initial_step)
    finally:
        worker._shutdown_runtime_dependencies()


def _run_video_process(args) -> None:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_video_process_main,
        args=(
            args.plan,
            args.step,
            args.video,
            args.video_dir,
            float(args.fps),
            bool(args.highpoly_stones),
            args.camera_center,
            args.camera_position,
        ),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise SystemExit(
            "Video rendering failed in the Open3D renderer subprocess "
            f"(exit code {proc.exitcode})."
        )


def main():
    args = _parse_args()
    if args.video is not None:
        _run_video_process(args)
        return

    worker = PlanReplayWorker(
        plan_id=args.plan,
        start_step=None,
        on_log=lambda msg: print(msg),
    )
    try:
        worker.setup_for_replay()
        if not worker.action_sequence:
            raise ValueError(f"No actions found in plan: {worker.plan_dir}")
        worker._stone_highpoly_mesh_mode = bool(args.highpoly_stones)
        initial_step = min(args.step, len(worker.action_sequence)) - 1
        PlanMotionViewer(
            worker,
            initial_step=initial_step,
            fps=args.fps,
            camera_center=args.camera_center,
            camera_position=args.camera_position,
        )
    finally:
        worker._shutdown_runtime_dependencies()


if __name__ == "__main__":
    main()
