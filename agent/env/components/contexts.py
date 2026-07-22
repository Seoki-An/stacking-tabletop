import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
DIFFSIM_PY_BUILD = ROOT_DIR.parent / "diffsim" / "interop" / "python" / "build"
if DIFFSIM_PY_BUILD.exists() and str(DIFFSIM_PY_BUILD) not in sys.path:
    sys.path.insert(0, str(DIFFSIM_PY_BUILD))

from diffsimpy import diffsim, posegen, poseinit, sceneid

from agent.config_views import support_config

_DIFFSIM_CONFIG_CACHE: dict[tuple[str, str], str] = {}


def environment_ground_height(cfg) -> float:
    try:
        return support_config(cfg).ground_z
    except Exception:
        return 0.0


def set_environment_ground_height(cfg, ground_height: float) -> None:
    if not hasattr(cfg, "reward"):
        return
    if not hasattr(cfg.reward, "support") or cfg.reward.support is None:
        cfg.reward.support = {}
    cfg.reward.support.ground_z = float(ground_height)


def _diffsim_base_config_path(fast: bool = False) -> str:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = "diffsim_fast.yml" if fast else "diffsim.yml"
    return os.path.join(cur_dir, "../../configs", cfg_file)


def _height_key(ground_height: float) -> str:
    return f"{float(ground_height):.9f}".replace("-", "m").replace(".", "p")


def _diffsim_config_path(
    fast: bool = False,
    ground_height: float | None = None,
) -> str:
    path = _diffsim_base_config_path(fast)
    if ground_height is None or abs(float(ground_height)) < 1e-9:
        return path

    key = (path, _height_key(float(ground_height)))
    if key in _DIFFSIM_CONFIG_CACHE:
        return _DIFFSIM_CONFIG_CACHE[key]

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    bodies = data.setdefault("body", [])
    if not bodies:
        bodies.append({})
    geometry = bodies[0].setdefault("geometry", [])
    plane = next(
        (
            geom
            for geom in geometry
            if str(geom.get("type", "")).strip().lower() == "plane"
        ),
        None,
    )
    if plane is None:
        plane = {"type": "plane", "mu": 1.0}
        geometry.insert(0, plane)
    plane["normal"] = [0.0, 0.0, 1.0]
    plane["center"] = [0.0, 0.0, float(ground_height)]

    tmp_path = (
        Path(tempfile.gettempdir()) / f"stacking_{Path(path).stem}_ground_{key[1]}.yml"
    )
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    _DIFFSIM_CONFIG_CACHE[key] = str(tmp_path)
    return str(tmp_path)


def get_diffsim(
    fast: bool = False,
    ground_height: float | None = None,
) -> Tuple[diffsim.Context, diffsim.BodyConfig]:
    path = _diffsim_config_path(fast, ground_height)
    return diffsim.Context(path), diffsim.BodyConfig(path)


def get_diffsim_plane_config(
    fast: bool = False,
    ground_height: float | None = None,
) -> diffsim.BodyConfig:
    return diffsim.BodyConfig(_diffsim_config_path(fast, ground_height))


def _set_posegen_ground_height_if_supported(
    config, ground_height: float | None
) -> None:
    if ground_height is None:
        return
    for obj in (getattr(config, "obj", None), config):
        if obj is None:
            continue
        for name in ("ground_height", "ground_z", "plane_height", "floor_height"):
            if hasattr(obj, name):
                setattr(obj, name, float(ground_height))
                return


def get_posegen(ground_height: float | None = None) -> posegen.Context:
    config = posegen.Config()
    _set_posegen_ground_height_if_supported(config, ground_height)

    context = posegen.Context(config)
    context.config().obj.eps_gap = 1e-3
    context.config().obj.eps_comp = 1e-3
    context.config().obj.eps_target = 0.005
    context.config().obj.k_box = 20.0
    context.config().obj.w_box = 2.0
    context.config().obj.narrow_phase_new_tol = 0.01
    return context

def get_sceneid(
    ground_height: float | None = None,
) -> Tuple[sceneid.Context, sceneid.Config]:
    config = sceneid.Config()

    config.n_threads = 0
    config.log_interval = 1

    config.tr.max_iter = 20
    config.tr.eps = 0.1
    config.tr.delta_init = 0.125

    config.graph.max_iter = 100

    config.obj.k_pcd = 5.0
    config.obj.pcd_huber_delta = 0.02
    config.obj.pcd_max_gap = 0.15
    config.obj.k_gap_c = 30.0
    config.obj.k_comp = 0.0
    if ground_height is not None and hasattr(config.obj, "ground_height"):
        config.obj.ground_height = float(ground_height)

    return sceneid.Context(config), config
