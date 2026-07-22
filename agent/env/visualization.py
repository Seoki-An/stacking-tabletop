import os

import imageio
from omegaconf import OmegaConf
import open3d as o3d

import numpy as np
from tqdm import tqdm
from typing import Dict, List

from .components.state import State, StoneTrajectory
from .components.inventory import InventoryManager, TargetWall
from .components.action import Action
from .components.stone import StoneObject
from utils import pose_to_transformation_matrix, SuppressOutput

N_STEP_INIT = 15
N_STEP_POSEGEN = 10

# Two symmetric diagonal views; each rendered at half width and combined
# horizontally so the output frame is 1440×720.
_VIDEO_CAMERAS = [
    {"eye": [3.0, 3.0, 2.5], "look_at": [0.0, 0.0, 0.8], "up": [0, 0, 1]},
    {"eye": [-3.0, 3.0, 2.5], "look_at": [0.0, 0.0, 0.8], "up": [0, 0, 1]},
]

_FIGURE_STONE_COLORS = np.array(
    [
        [0.34, 0.56, 0.86, 1.0],
        [0.25, 0.70, 0.38, 1.0],
        [0.90, 0.55, 0.24, 1.0],
        [0.64, 0.42, 0.82, 1.0],
        [0.86, 0.34, 0.40, 1.0],
        [0.30, 0.68, 0.72, 1.0],
    ],
    dtype=float,
)


def _render_two_view(
    renderer: o3d.visualization.rendering.OffscreenRenderer,
) -> np.ndarray:
    """Render front-left and front-right views, each at half width, side by side."""
    imgs = []
    for cam in _VIDEO_CAMERAS:
        renderer.scene.camera.look_at(cam["look_at"], cam["eye"], cam["up"])
        imgs.append(np.asarray(renderer.render_to_image())[:, ::2])
    return np.concatenate(imgs, axis=1)


class VisualInfo:
    def __init__(
        self,
        cfg: OmegaConf,
        state: State,
        inventory: InventoryManager,
    ):
        self.cfg = cfg
        self.state = state
        self.inventory = inventory


def save_configuration_figure(
    image_filename: str,
    visual_info: VisualInfo,
) -> None:
    """Save a static final stack figure without using Open3D offscreen rendering."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    os.makedirs(os.path.dirname(image_filename) or ".", exist_ok=True)

    target_wall = visual_info.inventory.target_wall
    state = visual_info.state

    target_meshes = [
        geometry.get_mesh_array() for geometry in target_wall.geometries
    ]
    stone_meshes = []
    all_points = [points for points, _ in target_meshes if len(points) > 0]

    for order, stone_idx in enumerate(state.stone_seq):
        stone_idx = int(stone_idx)
        if stone_idx < 0 or stone_idx >= len(visual_info.inventory.stones):
            continue
        stone = visual_info.inventory.stones[stone_idx]
        pose = state.stone_poses.get(int(stone.id), stone.pose)
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            continue

        points, faces = stone.get_dsf_mesh_array()
        transform = pose_to_transformation_matrix(pose)
        world_points = points @ transform[:3, :3].T + transform[:3, 3]
        color = _FIGURE_STONE_COLORS[order % len(_FIGURE_STONE_COLORS)]
        stone_meshes.append((world_points, faces, color))
        if len(world_points) > 0:
            all_points.append(world_points)

    if all_points:
        pts = np.concatenate(all_points, axis=0)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
    else:
        pts = np.zeros((1, 3), dtype=float)
    if len(pts) == 0:
        pts = np.zeros((1, 3), dtype=float)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-3)
    pad = max(float(span.max()) * 0.08, 0.15)
    xlim = (center[0] - 0.5 * span[0] - pad, center[0] + 0.5 * span[0] + pad)
    ylim = (center[1] - 0.5 * span[1] - pad, center[1] + 0.5 * span[1] + pad)
    zlim = (
        min(-0.05, mins[2] - pad),
        max(float(target_wall.height) + pad, maxs[2] + pad, 1.0),
    )

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    views = [("front-left", 24, -45), ("front-right", 24, 45)]
    for i, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        _plot_ground(ax, xlim, ylim, Poly3DCollection)
        for points, faces in target_meshes:
            _plot_mesh(
                ax,
                points,
                faces,
                color=(0.55, 0.62, 0.86, 0.22),
                edge_color=(0.40, 0.45, 0.65, 0.20),
                poly_cls=Poly3DCollection,
            )
        for points, faces, color in stone_meshes:
            _plot_mesh(
                ax,
                points,
                faces,
                color=color,
                edge_color=(0.10, 0.10, 0.10, 0.18),
                poly_cls=Poly3DCollection,
            )

        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(
            (
                max(xlim[1] - xlim[0], 1e-3),
                max(ylim[1] - ylim[0], 1e-3),
                max(zlim[1] - zlim[0], 1e-3),
            )
        )

    fig.suptitle(
        f"final configuration: {len(state.stone_seq)} stones",
        fontsize=13,
    )
    fig.savefig(image_filename, dpi=160)
    plt.close(fig)


def _plot_ground(ax, xlim, ylim, poly_cls) -> None:
    corners = [
        (xlim[0], ylim[0], 0.0),
        (xlim[1], ylim[0], 0.0),
        (xlim[1], ylim[1], 0.0),
        (xlim[0], ylim[1], 0.0),
    ]
    poly = poly_cls(
        [corners],
        facecolors=[(0.50, 0.58, 0.44, 0.14)],
        edgecolors=[(0.35, 0.40, 0.32, 0.18)],
        linewidths=0.4,
    )
    ax.add_collection3d(poly)


def _plot_mesh(
    ax,
    points: np.ndarray,
    faces: np.ndarray,
    color,
    edge_color,
    poly_cls,
) -> None:
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if points.ndim != 2 or points.shape[1] != 3:
        return
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        return

    finite_points = np.all(np.isfinite(points), axis=1)
    valid_faces = np.all((faces >= 0) & (faces < len(points)), axis=1)
    faces = faces[valid_faces]
    if len(faces) == 0:
        return
    faces = faces[np.all(finite_points[faces], axis=1)]
    if len(faces) == 0:
        return

    triangles = points[faces]
    poly = poly_cls(
        triangles,
        facecolors=[color],
        edgecolors=[edge_color],
        linewidths=0.15,
    )
    ax.add_collection3d(poly)


def visualization(
    video_filename: str,
    fps: float,
    time_scale: float,
    visual_info: VisualInfo,
):
    target_wall: TargetWall = visual_info.inventory.target_wall
    stone_seq: List[int] = visual_info.state.stone_seq
    stones: List[StoneObject] = [visual_info.inventory.stones[i] for i in stone_seq]
    action_history: List[Action] = visual_info.state.action_history
    trajectories: List[Dict[str, StoneTrajectory]] = visual_info.state.trajectories

    assert (
        len(stone_seq) == len(action_history) == len(trajectories)
    ), f"the length of stone_seq, action_history, and trajectories should be the same: {len(stone_seq)}, {len(action_history)}, {len(trajectories)}"

    dt = visual_info.cfg.sim.dt

    ### Plot simulated result ###
    with SuppressOutput():
        renderer = o3d.visualization.rendering.OffscreenRenderer(1440, 720)
        renderer.scene.camera.look_at([0, 0, 0], [3.0, 3.0, 3.0], [0, 0, 1])
        renderer.scene.scene.set_indirect_light_intensity(25000)

    frames = []

    # plot ground surface
    plane_width = target_wall.cfg.width + 10.0
    plane_length = target_wall.cfg.length + 10.0

    plane = o3d.t.geometry.TriangleMesh.create_box(plane_width, plane_length, 0.001)
    plane = plane.translate([-plane_width / 2, -plane_length / 2, -0.001])
    plane.compute_vertex_normals()
    mat_plane = o3d.visualization.rendering.MaterialRecord()
    mat_plane.base_color = np.array([0.5, 0.7, 0.3, 0.5])
    mat_plane.shader = "defaultLit"
    renderer.scene.add_geometry("ground_plane", plane, mat_plane)

    # plot target wall
    mat_target = o3d.visualization.rendering.MaterialRecord()
    mat_target.base_color = np.array([0.8, 0.8, 0.8, 0.2])
    mat_target.shader = "defaultLitTransparency"
    mat_target.has_alpha = True
    for i, geometry in enumerate(target_wall.geometries):
        renderer.scene.add_geometry(f"target_wall_{i}", geometry.get_mesh(), mat_target)

    simul_t = 0.0

    mat_stone_init = o3d.visualization.rendering.MaterialRecord()
    mat_stone_init.base_color = np.array([0.5, 0.5, 0.5, 0.5])

    mat_stone_init.base_color = np.array([0.5, 0.5, 0.5, 1.0])
    mat_stone_init.shader = "defaultLit"

    mat_stone_mid = o3d.visualization.rendering.MaterialRecord()
    mat_stone_mid.base_color = np.array([0.2, 1.0, 0.2, 0.5])
    mat_stone_mid.shader = "defaultLitTransparency"
    mat_stone_mid.has_alpha = True

    mat_stone_fin = o3d.visualization.rendering.MaterialRecord()
    mat_stone_fin.base_color = np.array([0.3, 0.3, 0.3, 1.0])
    mat_stone_fin.shader = "defaultLit"

    mat_stone_IoU = o3d.visualization.rendering.MaterialRecord()
    mat_stone_IoU.base_color = np.array([0.2, 0.9, 0.2, 0.8])
    mat_stone_IoU.shader = "defaultLitTransparency"
    mat_stone_IoU.has_alpha = True

    plotted_stone_models = {}

    for _ in range(N_STEP_INIT):
        frames.append(_render_two_view(renderer))

    for step in tqdm(range(len(stones))):
        id_step = stones[step].id
        trajectory = trajectories[step]

        stone_models_init = {}
        for i, geometry in enumerate(stones[step].geometries):
            stone_models_init[f"stone_{step}_{i}_init"] = geometry.get_mesh()
        plot_stone_model(
            action_history[step].init_pose,
            stone_models_init,
            renderer,
            mat_stone_init,
        )
        for _ in range(N_STEP_INIT):
            frames.append(_render_two_view(renderer))

        stone_models_mid = {}
        for i, geometry in enumerate(stones[step].geometries):
            stone_models_mid[f"stone_{step}_{i}_mid"] = geometry.get_mesh()
        plot_stone_model(
            action_history[step].pose,
            stone_models_mid,
            renderer,
            mat_stone_mid,
        )
        for _ in range(N_STEP_POSEGEN):
            frames.append(_render_two_view(renderer))

        for name in stone_models_init.keys():
            renderer.scene.remove_geometry(name)

        for _ in range(N_STEP_POSEGEN):
            frames.append(_render_two_view(renderer))

        for name in stone_models_mid.keys():
            renderer.scene.remove_geometry(name)

        stone_models_fin = {}
        for i, geometry in enumerate(stones[step].geometries):
            stone_models_fin[f"{id_step}_{i}"] = geometry.get_mesh()
        plot_stone_model(
            np.array([0, 0, 0, 0, 0, 0, 1.0]),
            stone_models_fin,
            renderer,
            mat_stone_fin,
        )
        plotted_stone_models[id_step] = [name for name in stone_models_fin.keys()]

        time_length = len(trajectory[id_step])
        time_interval = max(1, round(time_scale / (dt * fps)))

        for i in range(0, time_length, time_interval):
            simul_t += time_interval * dt

            for id, names in plotted_stone_models.items():
                stone_trajectory = trajectory.get(id)
                pose = trajectory_pose_at(stone_trajectory, i)
                if pose is None:
                    continue
                T = pose_to_transformation_matrix(pose)
                for name in names:
                    renderer.scene.set_geometry_transform(name, T)

            frames.append(_render_two_view(renderer))

        stone_IoU_models = {}
        for i, model in enumerate(stones[step].get_IoU_models()):
            stone_IoU_models[f"stone_IoU_{step}_{i}"] = model
        plot_stone_model(
            None,
            stone_IoU_models,
            renderer,
            mat_stone_IoU,
        )
        for i in range(N_STEP_POSEGEN):
            frames.append(_render_two_view(renderer))

        if len(stone_IoU_models) > 0:
            for name in stone_IoU_models.keys():
                renderer.scene.remove_geometry(name)

    imageio.mimsave(video_filename, frames, fps=fps)


def trajectory_pose_at(trajectory: StoneTrajectory, frame: int):
    """Return a frame pose, holding the last valid sample when needed."""
    if trajectory is None or len(trajectory) == 0:
        return None
    return trajectory[min(max(int(frame), 0), len(trajectory) - 1)]


def plot_stone_model(
    pose: np.ndarray,
    stone: Dict[str, o3d.geometry.TriangleMesh],
    renderer: o3d.visualization.rendering.OffscreenRenderer,
    mat: o3d.visualization.rendering.MaterialRecord = None,
):
    """
    Arguments:
        pose = [position, quaternion] numpy vector
        stone = Dict(stone mesh model)
        renderer = open3d renderer object
    """
    for name, mesh in stone.items():
        if pose is not None:
            mesh.transform(pose_to_transformation_matrix(pose))
        mesh.compute_vertex_normals()
        renderer.scene.add_geometry(name, mesh, mat)
