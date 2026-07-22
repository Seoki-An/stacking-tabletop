"""Worker for field execution of the stacking planner.

Planning, perception and ROS control run on a background thread.  The worker
communicates with the GUI exclusively through plain Python callbacks injected
at construction time, so it is decoupled from any GUI framework.

Heavy / process-altering imports (Ray, CUDA, rclpy) are deferred into the
worker thread; see PlanningWorker._load_runtime_dependencies.
"""

import copy
import datetime
from contextlib import contextmanager
import os
from pathlib import Path
import pickle
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from agent.env.components.action.floor_fill import (
    active_floor_context,
    active_layer_fill_metrics,
    lower_floor_fill_reject_stacked,
)
from utils.log_paths import (
    DESKTOP_LOG_SUFFIX,
    NUC_LOG_SUFFIX,
    equivalent_log_paths,
    unique_suffixed_dir,
    with_log_machine_suffix,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


VISUALIZATION_ON = False
PERCEPTION_ON = True
ROS_CONTROL_ON = True
LIVE_JOINT_VIEWER_ON = _env_flag("STACKING_LIVE_JOINT_VIEWER", True)
LIVE_JOINT_VIEWER_MIN_INTERVAL = float(
    os.environ.get("STACKING_LIVE_JOINT_VIEWER_MIN_INTERVAL", "1.0")
)

INHAND_REPLAN_TRANSLATION_THRESHOLD = 0.03  # meters
INHAND_REPLAN_ROTATION_THRESHOLD_DEG = 5.0
TARGET_STRUCTURE_VOXEL = 0.1  # meters
TARGET_STRUCTURE_ALPHA = 0.24
TARGET_STRUCTURE_PLACE_HEIGHT_BOX_MARGIN = 0.25
TARGET_STRUCTURE_PLACE_HEIGHT_BOX_MIN_THICKNESS = 0.04
INTERMEDIATE_PUTDOWN_MAX_ATTEMPTS = 40
EXECUTION_MAX_NUM_REGRASP_SOLUTIONS = 3
PLANNING_PREVIEW_MAX_MARKERS = 40
SCENE_SCAN_DONE_WAIT_TIMEOUT = 240.0
SCENE_PCD_DONE_DRAIN_SECONDS = 2.0
SCENE_PCD_MERGE_VOXEL = 0.003
SCENE_PCD_DISPLAY_VOXEL = 0.02
MANUAL_PLACE_PUBLISH_COUNT = 1000
MANUAL_PLACE_OPEN_PUBLISH_COUNT = 20000
PLACE_RELEASE_STEP_PUBLISH_COUNT = 1000

C_GROUND = [0.5, 0.7, 0.3]
C_EXCAV = [0.95, 0.82, 0.28]
C_STONE = [0.70, 0.70, 0.75]
C_TARGET = [0.85, 0.40, 0.20]
C_GRIPPER = [0.20, 0.70, 0.90]
C_GHOST = [0.85, 0.40, 0.20]
C_STRUCTURE = [0.30, 0.30, 0.55]
C_INHAND_INIT = [0.50, 0.50, 0.50]
C_INITIAL_TRAJECTORY = [0.15, 0.35, 0.95]
C_REPLANNED_TRAJECTORY = [0.95, 0.20, 0.15]
C_BASIN_RELEASE = [0.95, 0.55, 0.10]
C_STABLE_POSE = [0.10, 0.65, 0.35]
C_SCENEID = [0.10, 0.75, 0.95]
C_SCENEID_PRE = [1.00, 0.62, 0.10]
C_SCENEID_DELTA = [1.00, 0.95, 0.10]
C_SCENEID_GROUND = [0.05, 0.55, 0.95]
C_SCENE_SCAN_PCD = [0.12, 0.85, 0.55]
LIDAR_FRAME_LINK_NAMES = ("lidar1_link", "lidar2_link", "lidar3_link")
INHAND_PREVIEW_RIGHT_OFFSET = np.array([3.5, 0.0, 0.0])
Z_OFFSET_INTERMEDIATE = (
    0.10  # meters, for intermediate putdown to avoid collision with the ground
)
Z_OFFSET_PLACE = 0.10  # meters, to prevent planning failure due to slight intersection with the target structure
ONLINE_PLACE_Z_OFFSET = 0.20  # match scripts.desktop.generate_sequence.PLACE_Z_OFFSET
