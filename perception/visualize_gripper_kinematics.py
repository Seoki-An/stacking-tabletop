from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import inrol_urdf_parser as urdf
import numpy as np
import open3d as o3d

from model import MANIPULATOR_PATH, get_excavator_model, update_urdf_mesh

GRIPPER_LINKS = ("cs_tilt", "cs_rotate", "grip_body", "grip_left", "grip_right")
DEFAULT_LIDAR_LINK = "lidar2_link"
DEFAULT_GRIPPER_BODY_LINK = "grip_body"
CSV_JOINT_COLUMNS = tuple(f"joint_{i}" for i in range(6))
CSV_COMMAND_JOINT_COLUMNS = tuple(f"command_joint_{i}" for i in range(6))
CSV_POSE_GROUP_COLUMNS = (
    "pose_label",
    "bucket_offset",
    "tilt_offset",
    "rotation_index",
    "grab",
    "opening_angle",
)
JOINT_OFFSET_NAMES = ("swing", "boom", "arm", "bucket", "tilt", "rotate")
C_LIDAR = [0.15, 0.42, 1.0]
C_GRIPPER = [0.95, 0.58, 0.16]
C_MANIPULATOR = [0.55, 0.55, 0.55]
C_MODEL_PCD = [1.0, 0.16, 0.08]
C_ORIGINAL = [0.1, 0.1, 0.1]
FRAME_SIZE = 0.7
DEFAULT_MANIPULATOR_URDF = Path(MANIPULATOR_PATH) / "vdk23_cx.urdf"
DEFAULT_CALIBRATION_JOINT = "lidar2_joint"


@dataclass(frozen=True)
class GripperScanSample:
    index: int
    timestamp: int
    pcd_path: Path
    q6: np.ndarray
    q8: np.ndarray
    opening_angle: float
    row: dict[str, str]


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _parse_timestamp(value: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError("empty timestamp")
    try:
        return int(text)
    except ValueError:
        try:
            return int(Decimal(text))
        except InvalidOperation as exc:
            raise ValueError(f"invalid timestamp: {value!r}") from exc


def _resolve_scan_paths(
    scan_dir: Path, csv_path: str | None, pcd_dir: str | None
) -> tuple[Path, Path]:
    if csv_path is None:
        csv_resolved = scan_dir / "joint_log.csv"
    else:
        csv_resolved = Path(csv_path)
        if not csv_resolved.is_absolute():
            csv_resolved = scan_dir / csv_resolved

    if pcd_dir is None:
        pcd_resolved = scan_dir / "pcd"
    else:
        pcd_resolved = Path(pcd_dir)
        if not pcd_resolved.is_absolute():
            pcd_resolved = scan_dir / pcd_resolved

    return csv_resolved, pcd_resolved


def _pcd_map(pcd_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in pcd_dir.rglob("*.pcd"):
        try:
            paths[int(path.stem)] = path
        except ValueError:
            continue
    return paths


def load_gripper_scan_samples(
    scan_dir: str | Path,
    csv_path: str | None = None,
    pcd_dir: str | None = None,
    opening_angle: float | None = None,
    default_opening_angle: float = 0.0,
) -> list[GripperScanSample]:
    scan_dir = Path(scan_dir)
    csv_resolved, pcd_resolved = _resolve_scan_paths(scan_dir, csv_path, pcd_dir)
    if not csv_resolved.is_file():
        raise FileNotFoundError(f"joint CSV not found: {csv_resolved}")
    if not pcd_resolved.is_dir():
        raise FileNotFoundError(f"PCD directory not found: {pcd_resolved}")

    paths_by_time = _pcd_map(pcd_resolved)
    samples = []
    with csv_resolved.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_resolved} is empty")
        missing = [
            name for name in CSV_JOINT_COLUMNS if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"{csv_resolved} is missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            timestamp = _parse_timestamp(row["time"])
            pcd_path_match = paths_by_time.get(timestamp)
            if pcd_path_match is None:
                continue

            q6 = np.asarray([float(row[name]) for name in CSV_JOINT_COLUMNS])
            if opening_angle is None:
                q_left = _as_float(
                    row,
                    "gripper_left_joint",
                    _as_float(row, "opening_angle", default_opening_angle),
                )
                q_right = _as_float(
                    row,
                    "gripper_right_joint",
                    _as_float(row, "opening_angle", default_opening_angle),
                )
            else:
                q_left = q_right = float(opening_angle)
            q8 = np.concatenate([q6, [q_left, q_right]])
            samples.append(
                GripperScanSample(
                    index=len(samples),
                    timestamp=timestamp,
                    pcd_path=pcd_path_match,
                    q6=q6,
                    q8=q8,
                    opening_angle=float((q_left + q_right) * 0.5),
                    row=row,
                )
            )

    if not samples:
        raise ValueError(
            f"No CSV rows in {csv_resolved} had matching .pcd files under {pcd_resolved}"
        )
    return samples


def _row_text(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return "" if value is None else str(value).strip()


def _pose_group_key(sample: GripperScanSample) -> tuple[str, ...]:
    command_values = tuple(
        _row_text(sample.row, name) for name in CSV_COMMAND_JOINT_COLUMNS
    )
    pose_values = tuple(_row_text(sample.row, name) for name in CSV_POSE_GROUP_COLUMNS)
    if any(command_values):
        return ("command", *pose_values, *command_values)
    if any(pose_values):
        return ("metadata", *pose_values)
    return (
        "feedback",
        *(f"{value:.3f}" for value in sample.q6),
        f"{sample.opening_angle:.3f}",
    )


def representative_samples_by_pose(
    samples: list[GripperScanSample],
) -> list[GripperScanSample]:
    groups: dict[tuple[str, ...], list[GripperScanSample]] = {}
    for sample in samples:
        groups.setdefault(_pose_group_key(sample), []).append(sample)
    return [group[len(group) // 2] for group in groups.values()]


def sample_with_joint_offsets(
    sample: GripperScanSample,
    joint_offsets: Iterable[float],
) -> GripperScanSample:
    offsets = np.asarray(list(joint_offsets), dtype=np.float64)
    if offsets.shape != (6,):
        raise ValueError(f"Expected six joint offsets, got shape {offsets.shape}")
    q6 = sample.q6 + offsets
    q8 = sample.q8.copy()
    q8[:6] = q6
    return GripperScanSample(
        index=sample.index,
        timestamp=sample.timestamp,
        pcd_path=sample.pcd_path,
        q6=q6,
        q8=q8,
        opening_angle=sample.opening_angle,
        row=sample.row,
    )


def _parse_vector(text: str | None, default: Iterable[float]) -> np.ndarray:
    if text is None:
        return np.asarray(list(default), dtype=np.float64)
    values = [float(item) for item in text.split()]
    if len(values) != 3:
        return np.asarray(list(default), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _format_vector(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


class UrdfOriginEditor:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()
        self._joint_elements = {}
        self._baseline = {}

        for joint in self.root.findall(".//joint"):
            name = joint.get("name")
            if not name:
                continue
            origin = joint.find("origin")
            if origin is None:
                origin = ET.SubElement(joint, "origin")
                origin.set("xyz", "0 0 0")
                origin.set("rpy", "0 0 0")
            xyz = _parse_vector(origin.get("xyz"), [0.0, 0.0, 0.0])
            rpy = _parse_vector(origin.get("rpy"), [0.0, 0.0, 0.0])
            self._joint_elements[name] = origin
            self._baseline[name] = (xyz, rpy)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_elements.keys())

    def get_origin(self, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
        origin = self._joint_elements[joint_name]
        return (
            _parse_vector(origin.get("xyz"), [0.0, 0.0, 0.0]),
            _parse_vector(origin.get("rpy"), [0.0, 0.0, 0.0]),
        )

    def set_origin(
        self,
        joint_name: str,
        xyz: Iterable[float],
        rpy: Iterable[float],
    ) -> None:
        origin = self._joint_elements[joint_name]
        origin.set("xyz", _format_vector(xyz))
        origin.set("rpy", _format_vector(rpy))

    def reset_joint(self, joint_name: str) -> None:
        xyz, rpy = self._baseline[joint_name]
        self.set_origin(joint_name, xyz, rpy)

    def reset_all(self) -> None:
        for joint_name in self.joint_names:
            self.reset_joint(joint_name)

    def changed_origins(self, atol: float = 1e-12) -> dict[str, dict[str, list[float]]]:
        changed = {}
        for joint_name in self.joint_names:
            xyz, rpy = self.get_origin(joint_name)
            xyz0, rpy0 = self._baseline[joint_name]
            if np.allclose(xyz, xyz0, atol=atol) and np.allclose(rpy, rpy0, atol=atol):
                continue
            changed[joint_name] = {
                "xyz": xyz.tolist(),
                "rpy": rpy.tolist(),
                "delta_xyz": (xyz - xyz0).tolist(),
                "delta_rpy": (rpy - rpy0).tolist(),
            }
        return changed

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ET.indent(self.tree, space="    ")
        except AttributeError:
            pass
        self.tree.write(path, encoding="unicode", xml_declaration=True)
        return path

    def write_patch_json(self, path: str | Path, metadata: dict) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_urdf": str(self.urdf_path),
            "changed_origins": self.changed_origins(),
            **metadata,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def load_excavator_model_from_urdf(urdf_path: str | Path) -> urdf.URDF:
    model = urdf.URDF()
    ok = model.Parse(str(urdf_path))
    if ok is False:
        raise RuntimeError(f"Failed to parse URDF: {urdf_path}")
    return model


def _link_transform(model, link_name: str, use_visual_geom: bool = False) -> np.ndarray:
    T = np.eye(4)
    link = model.GetLink(link_name)
    if use_visual_geom and link.geoms:
        geom = link.geoms[0]
        T[:3, :3] = geom.GetRotation()
        T[:3, 3] = geom.GetPosition()
    else:
        T[:3, :3] = link.GetRotation()
        T[:3, 3] = link.GetPosition()
    return T


def _paint_copy(geometry, color):
    painted = copy.deepcopy(geometry)
    painted.paint_uniform_color(color)
    return painted


def _transform_copy(geometry, T: np.ndarray):
    transformed = copy.deepcopy(geometry)
    transformed.transform(T)
    return transformed


def _combined_aabb(geometries: Iterable[o3d.geometry.Geometry3D], margin: float):
    points = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            pts = np.asarray(geometry.vertices)
        elif isinstance(geometry, o3d.geometry.PointCloud):
            pts = np.asarray(geometry.points)
        else:
            continue
        if pts.size:
            points.append(pts)
    if not points:
        return None
    stacked = np.concatenate(points, axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(
        stacked.min(axis=0) - margin,
        stacked.max(axis=0) + margin,
    )


def _crop_to_aabb(
    pcd: o3d.geometry.PointCloud,
    aabb: o3d.geometry.AxisAlignedBoundingBox | None,
) -> o3d.geometry.PointCloud:
    if aabb is None or len(pcd.points) == 0:
        return pcd
    indices = aabb.get_point_indices_within_bounding_box(pcd.points)
    return pcd.select_by_index(indices)


def _sample_model_surface(
    meshes: Iterable[o3d.geometry.TriangleMesh],
    points_per_mesh: int,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    for mesh in meshes:
        if len(mesh.triangles) == 0:
            continue
        pcd += mesh.sample_points_uniformly(number_of_points=points_per_mesh)
    return pcd


def _fk_link_meshes(
    sample: GripperScanSample,
    model,
    link_meshes: dict[str, o3d.geometry.TriangleMesh],
    lidar_link: str,
    frame: str,
    link_names: Iterable[str],
) -> list[o3d.geometry.TriangleMesh]:
    model.SetState(sample.q8.copy())
    lidar_T = _link_transform(model, lidar_link)
    frame_T = np.linalg.inv(lidar_T) if frame == "lidar" else np.eye(4)
    meshes_base = update_urdf_mesh(model, link_meshes, sample.q8.copy())
    return [
        _transform_copy(meshes_base[name], frame_T)
        for name in link_names
        if name in meshes_base
    ]


def _distance_summary(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> dict[str, float] | None:
    if len(source.points) == 0 or len(target.points) == 0:
        return None
    distances = np.asarray(source.compute_point_cloud_distance(target))
    if distances.size == 0:
        return None
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95.0)),
        "max": float(np.max(distances)),
    }


def _format_summary(name: str, summary: dict[str, float] | None) -> str:
    if summary is None:
        return f"{name}: n/a"
    return (
        f"{name}: mean={summary['mean']:.4f}m, "
        f"median={summary['median']:.4f}m, "
        f"p95={summary['p95']:.4f}m, max={summary['max']:.4f}m"
    )


def build_sample_geometries(
    sample: GripperScanSample,
    model,
    link_meshes: dict[str, o3d.geometry.TriangleMesh],
    lidar_link: str = DEFAULT_LIDAR_LINK,
    frame: str = "lidar",
    crop_to_gripper: bool = True,
    crop_margin: float = 0.35,
    voxel_size: float = 0.0,
    show_excavator: bool = False,
    show_frames: bool = True,
    model_points_per_mesh: int = 0,
) -> tuple[list[o3d.geometry.Geometry3D], dict[str, object]]:
    model.SetState(sample.q8.copy())
    lidar_T = _link_transform(model, lidar_link)
    frame_T = np.linalg.inv(lidar_T) if frame == "lidar" else np.eye(4)

    raw_pcd = o3d.io.read_point_cloud(str(sample.pcd_path))
    lidar_pcd = _transform_copy(raw_pcd, lidar_T if frame == "base" else np.eye(4))
    if voxel_size > 0:
        lidar_pcd = lidar_pcd.voxel_down_sample(voxel_size)

    meshes_base = update_urdf_mesh(model, link_meshes, sample.q8.copy())
    gripper_meshes = [
        _transform_copy(meshes_base[name], frame_T)
        for name in GRIPPER_LINKS
        if name in meshes_base
    ]
    excavator_meshes = [
        _transform_copy(mesh, frame_T)
        for name, mesh in meshes_base.items()
        if name not in GRIPPER_LINKS
    ]

    gripper_aabb = _combined_aabb(gripper_meshes, crop_margin)
    lidar_cropped = (
        _crop_to_aabb(lidar_pcd, gripper_aabb) if crop_to_gripper else lidar_pcd
    )

    geometries: list[o3d.geometry.Geometry3D] = []
    geometries.append(_paint_copy(lidar_cropped, C_LIDAR))
    geometries.extend(_paint_copy(mesh, C_GRIPPER) for mesh in gripper_meshes)
    if show_excavator:
        geometries.extend(_paint_copy(mesh, C_MANIPULATOR) for mesh in excavator_meshes)

    model_pcd = None
    lidar_to_model = None
    model_to_lidar = None
    if model_points_per_mesh > 0:
        model_pcd = _sample_model_surface(gripper_meshes, model_points_per_mesh)
        model_pcd.paint_uniform_color(C_MODEL_PCD)
        geometries.append(model_pcd)
        lidar_to_model = _distance_summary(lidar_cropped, model_pcd)
        model_to_lidar = _distance_summary(model_pcd, lidar_cropped)

    if show_frames:
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_SIZE)
        geometries.append(origin)
        if frame == "base":
            lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=FRAME_SIZE
            )
            lidar_frame.transform(lidar_T)
            geometries.append(lidar_frame)
        grip_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_SIZE)
        grip_frame.transform(
            frame_T @ _link_transform(model, DEFAULT_GRIPPER_BODY_LINK, True)
        )
        geometries.append(grip_frame)

    info = {
        "raw_points": len(raw_pcd.points),
        "shown_points": len(lidar_cropped.points),
        "lidar_to_model": lidar_to_model,
        "model_to_lidar": model_to_lidar,
        "q6": sample.q6.copy(),
        "q8": sample.q8.copy(),
        "opening_angle": sample.opening_angle,
        "timestamp": sample.timestamp,
        "pcd_path": sample.pcd_path,
    }
    return geometries, info


def _selected_samples(
    samples: list[GripperScanSample],
    sample_index: int | None,
    stride: int,
    max_samples: int | None,
) -> list[GripperScanSample]:
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(samples):
            raise IndexError(
                f"--sample-index {sample_index} is out of range for {len(samples)} samples"
            )
        return [samples[sample_index]]
    selected = samples[:: max(stride, 1)]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def _print_info(sample: GripperScanSample, info: dict[str, object]) -> None:
    print(
        f"sample={sample.index} time={info['timestamp']} "
        f"raw_points={info['raw_points']} shown_points={info['shown_points']} "
        f"opening_angle={info['opening_angle']:.4f}"
    )
    print("q6:", np.array2string(info["q6"], precision=4, suppress_small=True))
    print(_format_summary("lidar->model", info["lidar_to_model"]))
    print(_format_summary("model->lidar", info["model_to_lidar"]))


def _write_export(path: Path, geometries: list[o3d.geometry.Geometry3D]) -> None:
    pcd = o3d.geometry.PointCloud()
    mesh = o3d.geometry.TriangleMesh()
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.PointCloud):
            pcd += geometry
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            mesh += geometry

    suffix = path.suffix.lower()
    if (
        suffix in {".pcd", ".ply", ".xyz", ".xyzn", ".xyzrgb"}
        and len(pcd.points) > 0
    ):
        o3d.io.write_point_cloud(str(path), pcd)
        return
    if (
        suffix in {".ply", ".stl", ".obj", ".off", ".gltf", ".glb"}
        and len(mesh.vertices) > 0
    ):
        o3d.io.write_triangle_mesh(str(path), mesh)
        return
    raise ValueError(f"Cannot export geometries to {path}")


def _default_calibration_output_paths(scan_dir: str | Path) -> tuple[Path, Path]:
    root = Path(scan_dir)
    return (
        root / "calibrated_vdk23_cx.urdf",
        root / "gripper_kinematics_calibration.json",
    )


def run_calibration_gui(
    args: argparse.Namespace,
    samples: list[GripperScanSample],
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    from gui.viewer import Open3DViewer

    class CalibrationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Gripper Kinematics Calibration")
            self.resize(860, 620)

            self.samples = samples
            self.sample_index = int(np.clip(args.sample_index, 0, len(samples) - 1))
            self.frame = args.frame
            self.lidar_link = args.lidar_link
            self.crop_to_gripper = not args.no_crop
            self.crop_margin = args.crop_margin
            self.voxel_size = args.voxel_size
            self.model_points_per_mesh = args.model_points_per_mesh
            self.show_excavator = bool(args.show_excavator)

            self.output_urdf_path, self.output_json_path = (
                _default_calibration_output_paths(args.scan_dir)
            )
            if args.calibration_output_urdf is not None:
                self.output_urdf_path = Path(args.calibration_output_urdf)
            if args.calibration_output_json is not None:
                self.output_json_path = Path(args.calibration_output_json)

            self.editor = UrdfOriginEditor(args.calibration_urdf)
            self.temp_urdf_path = (
                Path(tempfile.gettempdir())
                / f"gripper_kinematics_calibration_{id(self)}.urdf"
            )
            _, self.link_meshes = get_excavator_model()
            self.base_model = load_excavator_model_from_urdf(args.calibration_urdf)
            self.current_model = None
            self.viewer = Open3DViewer("Gripper FK Calibration")
            self._syncing_controls = False
            self._syncing_joint_offsets = False

            central = QWidget(self)
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            top_row = QHBoxLayout()
            self.prev_btn = QPushButton("Prev")
            self.next_btn = QPushButton("Next")
            self.sample_label = QLabel()
            self.sample_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            top_row.addWidget(self.prev_btn)
            top_row.addWidget(self.next_btn)
            top_row.addWidget(self.sample_label, 1)
            layout.addLayout(top_row)

            options_row = QHBoxLayout()
            self.show_original_checkbox = QCheckBox("Original FK")
            self.show_original_checkbox.setChecked(True)
            self.show_frames_checkbox = QCheckBox("Frames")
            self.show_frames_checkbox.setChecked(True)
            self.show_excavator_checkbox = QCheckBox("Excavator")
            self.show_excavator_checkbox.setChecked(self.show_excavator)
            self.crop_checkbox = QCheckBox("Crop")
            self.crop_checkbox.setChecked(self.crop_to_gripper)
            for checkbox in (
                self.show_original_checkbox,
                self.show_frames_checkbox,
                self.show_excavator_checkbox,
                self.crop_checkbox,
            ):
                options_row.addWidget(checkbox)
            options_row.addStretch(1)
            layout.addLayout(options_row)

            joint_box = QGroupBox("URDF Joint Origin")
            joint_layout = QVBoxLayout(joint_box)
            joint_selector_row = QHBoxLayout()
            joint_selector_row.addWidget(QLabel("Joint"))
            self.joint_combo = QComboBox()
            self.joint_combo.addItems(self.editor.joint_names)
            if args.calibration_joint in self.editor.joint_names:
                self.joint_combo.setCurrentText(args.calibration_joint)
            elif DEFAULT_CALIBRATION_JOINT in self.editor.joint_names:
                self.joint_combo.setCurrentText(DEFAULT_CALIBRATION_JOINT)
            joint_selector_row.addWidget(self.joint_combo, 1)
            joint_layout.addLayout(joint_selector_row)

            spin_grid = QGridLayout()
            self.spin_boxes: dict[str, QDoubleSpinBox] = {}
            specs = [
                ("x", "m", -10.0, 10.0, args.calibration_pos_step),
                ("y", "m", -10.0, 10.0, args.calibration_pos_step),
                ("z", "m", -10.0, 10.0, args.calibration_pos_step),
                ("roll", "rad", -4.0 * np.pi, 4.0 * np.pi, args.calibration_rpy_step),
                ("pitch", "rad", -4.0 * np.pi, 4.0 * np.pi, args.calibration_rpy_step),
                ("yaw", "rad", -4.0 * np.pi, 4.0 * np.pi, args.calibration_rpy_step),
            ]
            for col, (name, suffix, min_value, max_value, step) in enumerate(specs):
                label = QLabel(name)
                spin = QDoubleSpinBox()
                spin.setRange(min_value, max_value)
                spin.setDecimals(6)
                spin.setSingleStep(step)
                spin.setSuffix(f" {suffix}")
                self.spin_boxes[name] = spin
                spin_grid.addWidget(label, 0, col)
                spin_grid.addWidget(spin, 1, col)
            joint_layout.addLayout(spin_grid)
            layout.addWidget(joint_box)

            offset_box = QGroupBox("Joint Offsets")
            offset_grid = QGridLayout(offset_box)
            self.joint_offset_boxes: dict[str, QDoubleSpinBox] = {}
            for col, name in enumerate(JOINT_OFFSET_NAMES):
                label = QLabel(name)
                spin = QDoubleSpinBox()
                spin.setRange(-np.pi, np.pi)
                spin.setDecimals(6)
                spin.setSingleStep(args.joint_offset_step)
                spin.setSuffix(" rad")
                self.joint_offset_boxes[name] = spin
                offset_grid.addWidget(label, 0, col)
                offset_grid.addWidget(spin, 1, col)
            layout.addWidget(offset_box)

            button_row = QHBoxLayout()
            self.reset_joint_btn = QPushButton("Reset Joint")
            self.reset_all_btn = QPushButton("Reset All")
            self.reset_offsets_btn = QPushButton("Reset Offsets")
            self.save_urdf_btn = QPushButton("Save URDF")
            self.save_json_btn = QPushButton("Save JSON")
            for button in (
                self.reset_joint_btn,
                self.reset_all_btn,
                self.reset_offsets_btn,
                self.save_urdf_btn,
                self.save_json_btn,
            ):
                button_row.addWidget(button)
            button_row.addStretch(1)
            layout.addLayout(button_row)

            self.origin_label = QLabel()
            self.origin_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.origin_label.setWordWrap(True)
            layout.addWidget(self.origin_label)

            self.metrics_label = QLabel()
            self.metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metrics_label.setWordWrap(True)
            layout.addWidget(self.metrics_label)

            self.save_label = QLabel(
                f"URDF: {self.output_urdf_path}\nJSON: {self.output_json_path}"
            )
            self.save_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.save_label.setWordWrap(True)
            layout.addWidget(self.save_label)

            self.prev_btn.clicked.connect(lambda: self._change_sample(-1))
            self.next_btn.clicked.connect(lambda: self._change_sample(1))
            self.joint_combo.currentTextChanged.connect(lambda *_: self._load_joint())
            for spin in self.spin_boxes.values():
                spin.valueChanged.connect(lambda *_: self._sync_from_spins())
            for spin in self.joint_offset_boxes.values():
                spin.valueChanged.connect(lambda *_: self._sync_from_joint_offsets())
            self.reset_joint_btn.clicked.connect(self._reset_joint)
            self.reset_all_btn.clicked.connect(self._reset_all)
            self.reset_offsets_btn.clicked.connect(self._reset_joint_offsets)
            self.save_urdf_btn.clicked.connect(self._save_urdf)
            self.save_json_btn.clicked.connect(self._save_json)
            for checkbox in (
                self.show_original_checkbox,
                self.show_frames_checkbox,
                self.show_excavator_checkbox,
                self.crop_checkbox,
            ):
                checkbox.toggled.connect(lambda *_: self._update_viewer())

            self._load_joint()

        def _sample(self) -> GripperScanSample:
            return self.samples[self.sample_index]

        def _joint_offsets(self) -> np.ndarray:
            return np.asarray(
                [self.joint_offset_boxes[name].value() for name in JOINT_OFFSET_NAMES],
                dtype=np.float64,
            )

        def _load_joint(self) -> None:
            joint_name = self.joint_combo.currentText()
            if not joint_name:
                return
            xyz, rpy = self.editor.get_origin(joint_name)
            self._syncing_controls = True
            values = {
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "roll": rpy[0],
                "pitch": rpy[1],
                "yaw": rpy[2],
            }
            for name, value in values.items():
                self.spin_boxes[name].setValue(float(value))
            self._syncing_controls = False
            self._update_viewer()

        def _sync_from_spins(self) -> None:
            if self._syncing_controls:
                return
            joint_name = self.joint_combo.currentText()
            if not joint_name:
                return
            xyz = np.asarray(
                [
                    self.spin_boxes["x"].value(),
                    self.spin_boxes["y"].value(),
                    self.spin_boxes["z"].value(),
                ],
                dtype=np.float64,
            )
            rpy = np.asarray(
                [
                    self.spin_boxes["roll"].value(),
                    self.spin_boxes["pitch"].value(),
                    self.spin_boxes["yaw"].value(),
                ],
                dtype=np.float64,
            )
            self.editor.set_origin(joint_name, xyz, rpy)
            self._update_viewer()

        def _sync_from_joint_offsets(self) -> None:
            if self._syncing_joint_offsets:
                return
            self._update_viewer()

        def _reload_model(self):
            self.editor.write(self.temp_urdf_path)
            self.current_model = load_excavator_model_from_urdf(self.temp_urdf_path)
            return self.current_model

        def _update_viewer(self) -> None:
            raw_sample = self._sample()
            joint_offsets = self._joint_offsets()
            sample = sample_with_joint_offsets(raw_sample, joint_offsets)
            model = self._reload_model()
            geometries, info = build_sample_geometries(
                sample,
                model,
                self.link_meshes,
                lidar_link=self.lidar_link,
                frame=self.frame,
                crop_to_gripper=self.crop_checkbox.isChecked(),
                crop_margin=self.crop_margin,
                voxel_size=self.voxel_size,
                show_excavator=self.show_excavator_checkbox.isChecked(),
                show_frames=self.show_frames_checkbox.isChecked(),
                model_points_per_mesh=self.model_points_per_mesh,
            )

            if self.show_original_checkbox.isChecked():
                original_meshes = _fk_link_meshes(
                    raw_sample,
                    self.base_model,
                    self.link_meshes,
                    self.lidar_link,
                    self.frame,
                    GRIPPER_LINKS,
                )
                geometries.extend(
                    (mesh, C_ORIGINAL, "wireframe") for mesh in original_meshes
                )

            self.viewer.set_geometries(geometries)
            self.sample_label.setText(
                f"sample {self.sample_index + 1}/{len(self.samples)}  "
                f"raw={raw_sample.index}  time={raw_sample.timestamp}  "
                f"pcd={raw_sample.pcd_path.name}"
            )
            self.metrics_label.setText(
                f"shown/raw points: {info['shown_points']}/{info['raw_points']}\n"
                f"{_format_summary('lidar->model', info['lidar_to_model'])}\n"
                f"{_format_summary('model->lidar', info['model_to_lidar'])}\n"
                "q6 raw: "
                + np.array2string(raw_sample.q6, precision=4, suppress_small=True)
                + "\noffset: "
                + np.array2string(joint_offsets, precision=4, suppress_small=True)
                + "\nq6 used: "
                + np.array2string(sample.q6, precision=4, suppress_small=True)
            )
            self._update_origin_label()

        def _update_origin_label(self) -> None:
            joint_name = self.joint_combo.currentText()
            if not joint_name:
                return
            xyz, rpy = self.editor.get_origin(joint_name)
            changed = self.editor.changed_origins()
            self.origin_label.setText(
                f"{joint_name} xyz={_format_vector(xyz)} "
                f"rpy={_format_vector(rpy)}\n"
                f"changed joints: {', '.join(changed.keys()) if changed else 'none'}"
            )

        def _change_sample(self, delta: int) -> None:
            self.sample_index = int(
                np.clip(self.sample_index + delta, 0, len(self.samples) - 1)
            )
            self._update_viewer()

        def _reset_joint(self) -> None:
            joint_name = self.joint_combo.currentText()
            if not joint_name:
                return
            self.editor.reset_joint(joint_name)
            self._load_joint()

        def _reset_all(self) -> None:
            self.editor.reset_all()
            self._load_joint()

        def _reset_joint_offsets(self) -> None:
            self._syncing_joint_offsets = True
            for spin in self.joint_offset_boxes.values():
                spin.setValue(0.0)
            self._syncing_joint_offsets = False
            self._update_viewer()

        def _patch_metadata(self) -> dict:
            raw_sample = self._sample()
            joint_offsets = self._joint_offsets()
            adjusted_sample = sample_with_joint_offsets(raw_sample, joint_offsets)
            return {
                "active_sample_index": self.sample_index,
                "active_sample_raw_index": raw_sample.index,
                "active_sample_timestamp": raw_sample.timestamp,
                "active_q6_raw": raw_sample.q6.tolist(),
                "active_q8_raw": raw_sample.q8.tolist(),
                "active_q6_adjusted": adjusted_sample.q6.tolist(),
                "active_q8_adjusted": adjusted_sample.q8.tolist(),
                "joint_offsets": {
                    name: float(value)
                    for name, value in zip(JOINT_OFFSET_NAMES, joint_offsets)
                },
                "frame": self.frame,
                "lidar_link": self.lidar_link,
            }

        def _save_urdf(self) -> None:
            path = self.editor.write(self.output_urdf_path)
            self.save_label.setText(f"Saved URDF: {path}\nJSON: {self.output_json_path}")

        def _save_json(self) -> None:
            path = self.editor.write_patch_json(
                self.output_json_path,
                self._patch_metadata(),
            )
            self.save_label.setText(f"URDF: {self.output_urdf_path}\nSaved JSON: {path}")

        def closeEvent(self, event):
            self.viewer.close()
            try:
                self.temp_urdf_path.unlink(missing_ok=True)
            except Exception:
                pass
            super().closeEvent(event)

    app = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    window.show()
    app.exec()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize gripper forward kinematics against LiDAR-2 scan data "
            "collected by scripts/nuc/scan_gripper_pcd.py."
        )
    )
    parser.add_argument(
        "scan_dir",
        help="Directory containing joint_log.csv and pcd/",
    )
    parser.add_argument("--csv", default=None, help="CSV path relative to scan_dir.")
    parser.add_argument(
        "--pcd-dir",
        default=None,
        help="PCD directory relative to scan_dir.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to visualize. Use --all to step through multiple samples.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Step through selected samples instead of showing only --sample-index.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride used with --all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to show when --all is set.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help=(
            "Use every matched CSV/PCD sample instead of one representative per "
            "commanded scan pose."
        ),
    )
    parser.add_argument(
        "--opening-angle",
        type=float,
        default=None,
        help="Override gripper left/right opening angle in radians.",
    )
    parser.add_argument(
        "--default-opening-angle",
        type=float,
        default=0.0,
        help="Opening angle used if the CSV has no gripper/opening columns.",
    )
    parser.add_argument(
        "--lidar-link",
        default=DEFAULT_LIDAR_LINK,
        help="URDF link for the raw PCD frame.",
    )
    parser.add_argument(
        "--frame",
        choices=["lidar", "base"],
        default="lidar",
        help="Visualization frame. 'lidar' overlays FK in the raw sensor frame.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Show the whole LiDAR cloud instead of points near the FK gripper AABB.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.35,
        help="AABB margin around the FK gripper when cropping LiDAR points.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel size for downsampling the LiDAR cloud.",
    )
    parser.add_argument(
        "--show-manipulator",
        action="store_true",
        help="Show non-gripper manipulator meshes as context.",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Hide origin, gripper, and LiDAR coordinate frames.",
    )
    parser.add_argument(
        "--model-points-per-mesh",
        type=int,
        default=1500,
        help="Sampled FK gripper surface points per mesh for distance summaries.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print FK-vs-LiDAR distance summaries without opening Open3D windows.",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Write the selected sample geometries to a PCD/PLY-style file.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Open an interactive URDF joint-origin calibration GUI.",
    )
    parser.add_argument(
        "--calibration-urdf",
        default=str(DEFAULT_MANIPULATOR_URDF),
        help="Excavator URDF to edit in calibration mode.",
    )
    parser.add_argument(
        "--calibration-joint",
        default=DEFAULT_CALIBRATION_JOINT,
        help="Initial URDF joint selected in calibration mode.",
    )
    parser.add_argument(
        "--calibration-output-urdf",
        default=None,
        help="Output path for the calibrated URDF.",
    )
    parser.add_argument(
        "--calibration-output-json",
        default=None,
        help="Output path for the calibration JSON patch.",
    )
    parser.add_argument(
        "--calibration-pos-step",
        type=float,
        default=0.005,
        help="Position spin-box step in meters for calibration mode.",
    )
    parser.add_argument(
        "--calibration-rpy-step",
        type=float,
        default=0.005,
        help="RPY spin-box step in radians for calibration mode.",
    )
    parser.add_argument(
        "--joint-offset-step",
        type=float,
        default=0.005,
        help="Joint-offset spin-box step in radians for calibration mode.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.calibration_pos_step <= 0.0:
        raise ValueError("--calibration-pos-step must be positive")
    if args.calibration_rpy_step <= 0.0:
        raise ValueError("--calibration-rpy-step must be positive")
    if args.joint_offset_step <= 0.0:
        raise ValueError("--joint-offset-step must be positive")

    raw_samples = load_gripper_scan_samples(
        args.scan_dir,
        csv_path=args.csv,
        pcd_dir=args.pcd_dir,
        opening_angle=args.opening_angle,
        default_opening_angle=args.default_opening_angle,
    )
    samples = (
        raw_samples
        if args.all_samples
        else representative_samples_by_pose(raw_samples)
    )
    if not args.all_samples and len(samples) != len(raw_samples):
        print(
            f"Using {len(samples)} representative poses from "
            f"{len(raw_samples)} matched CSV/PCD samples "
            "(pass --all-samples to keep every frame)."
        )

    if args.calibrate:
        run_calibration_gui(args, samples)
        return

    selected = _selected_samples(
        samples,
        sample_index=None if args.all else args.sample_index,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    model, link_meshes = get_excavator_model()

    last_geometries = None
    for sample in selected:
        geometries, info = build_sample_geometries(
            sample,
            model,
            link_meshes,
            lidar_link=args.lidar_link,
            frame=args.frame,
            crop_to_gripper=not args.no_crop,
            crop_margin=args.crop_margin,
            voxel_size=args.voxel_size,
            show_excavator=args.show_excavator,
            show_frames=not args.no_frames,
            model_points_per_mesh=args.model_points_per_mesh,
        )
        _print_info(sample, info)
        last_geometries = geometries
        if not args.metrics_only:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=(
                    f"Gripper FK vs LiDAR sample {sample.index} "
                    f"({args.frame} frame)"
                ),
            )

    if args.export is not None:
        if last_geometries is None:
            raise RuntimeError("No geometries were built for export")
        export_path = Path(args.export)
        _write_export(export_path, last_geometries)
        print(f"Exported selected geometries to {export_path}")


if __name__ == "__main__":
    main()
