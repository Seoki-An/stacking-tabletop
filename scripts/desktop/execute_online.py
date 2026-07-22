#!/usr/bin/env python3
"""Integrated desktop entry-point for the stacking planner.

When --plan is omitted, this starts a live online MCTS execution session.  Each
step is planned, shown in the GUI trajectory viewer, executed through ROS, and
then followed by perception before the next MCTS step is planned.  When --plan
is provided, the plan is copied into a branch and used as a per-step seed for
the same online replanning loop.

The Open3D GUI runs in the main process. Execution, ROS, Ray, and diffsim run
in a separate process so long-running native calls do not stall GUI event
handling.
"""

import argparse
import datetime
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import threading

import numpy as np

DISPLAY_QUEUE_MAXSIZE = 1
LIVE_QUEUE_MAXSIZE = 1
PLAN_BRANCH_SKIP_NAMES = {
    "__pycache__",
}
PLAN_BRANCH_SKIP_SUFFIXES = {
    ".pyc",
}
ONLINE_PLAN_SUFFIX = "online"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run live online MCTS planning and execution."
    )
    p.add_argument(
        "--plan",
        type=str,
        default=None,
        help=(
            "Existing plan identifier to branch and use as an online MCTS "
            "seed, e.g. '260606_1' for sessions/260606/plan_260606_1/. "
            "If omitted, execute_online.py starts a new online MCTS execution session."
        ),
    )
    p.add_argument(
        "--start_step",
        type=int,
        default=None,
        help=(
            "1-based action step to start physical execution. Defaults to the "
            "plan metadata execution_start_step for resumed branches, then 1."
        ),
    )
    p.add_argument(
        "--live_joint_viewer",
        dest="live_joint_viewer",
        action="store_true",
        default=True,
        help="Enable live 3D robot updates during control. Enabled by default.",
    )
    p.add_argument(
        "--no_live_joint_viewer",
        dest="live_joint_viewer",
        action="store_false",
        help="Disable live 3D robot updates during control.",
    )
    p.add_argument(
        "--live_joint_interval",
        type=float,
        default=1.0,
        help="Minimum seconds between live 3D joint updates.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="YAML",
        help=(
            "Config YAML for the online MCTS session. With --plan, this "
            "overrides the copied branch config while preserving the source "
            "plan as seed artifacts."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PLAN_DIR",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--resume-state-pkl",
        type=str,
        default=None,
        metavar="PKL",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--resume-start-step",
        type=int,
        default=None,
        metavar="STEP",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf override passed to online MCTS planning. May be repeated.",
    )
    p.add_argument(
        "--target-structure-offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Target structure XY offset for online MCTS planning.",
    )
    p.add_argument(
        "--regrasp-xy-pos",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Base regrasp XY position for online MCTS planning.",
    )
    p.add_argument(
        "--move-control-error-tol",
        type=float,
        default=None,
        help="Position-control error tolerance for move-path waypoints.",
    )
    p.add_argument(
        "--move-control-convergence-time-limit",
        type=float,
        default=None,
        help="Seconds allowed for final move-phase convergence.",
    )
    args = p.parse_args()
    if args.start_step is not None and args.start_step < 1:
        p.error("--start_step must be >= 1")
    if args.resume_state_pkl is not None and args.resume is None:
        p.error("--resume-state-pkl requires --resume")
    if args.resume_start_step is not None:
        if args.resume is None:
            p.error("--resume-start-step requires --resume")
        if args.resume_start_step < 1:
            p.error("--resume-start-step must be >= 1")
    if args.plan is not None:
        branch_conflict_options = [
            args.resume,
            args.resume_state_pkl,
            args.resume_start_step,
            args.target_structure_offset,
            args.regrasp_xy_pos,
        ]
        if any(value is not None for value in branch_conflict_options):
            p.error(
                "--resume, --target-structure-offset, and --regrasp-xy-pos "
                "cannot be used with --plan"
            )
    elif args.resume is not None:
        p.error(
            "online resume is not implemented in execute_online.py yet; use --plan to "
            "execute an existing saved plan"
        )
    elif args.start_step is not None:
        p.error("--start_step is only valid with --plan")
    for override in args.config_override:
        if "=" not in override:
            p.error(f"--config-override expects KEY=VALUE, got {override!r}")
    if args.move_control_error_tol is not None and (
        not np.isfinite(args.move_control_error_tol) or args.move_control_error_tol <= 0
    ):
        p.error("--move-control-error-tol must be > 0")
    if (
        args.move_control_convergence_time_limit is not None
        and (
            not np.isfinite(args.move_control_convergence_time_limit)
            or args.move_control_convergence_time_limit <= 0
        )
    ):
        p.error("--move-control-convergence-time-limit must be > 0")
    return args


def _build_control_options(args):
    options = {}
    if args.move_control_error_tol is not None:
        options["move_control_error_tol"] = float(args.move_control_error_tol)
    if args.move_control_convergence_time_limit is not None:
        options["move_control_convergence_time_limit"] = float(
            args.move_control_convergence_time_limit
        )
    return options


def _build_online_options(args):
    online_options = {
        "config": args.config,
        "config_override": list(args.config_override),
    }
    if args.config is not None:
        online_options["config"] = args.config
    if args.target_structure_offset is not None:
        online_options["target_structure_offset"] = list(
            args.target_structure_offset
        )
    if args.regrasp_xy_pos is not None:
        online_options["regrasp_xy_pos"] = list(args.regrasp_xy_pos)
    return online_options


def _plan_date_from_name(plan_name: str) -> str | None:
    parts = plan_name.split("_")
    if len(parts) >= 3 and parts[0] == "plan" and parts[1].isdigit():
        return parts[1]
    return None


def _resolve_plan_dir(plan: str) -> str:
    path = os.path.normpath(os.path.expanduser(plan))
    candidates = [path]

    plan_name = os.path.basename(path)
    parent = os.path.dirname(path)
    if not plan_name.startswith("plan_"):
        plan_name = f"plan_{plan_name}"
    date_str = _plan_date_from_name(plan_name)
    if date_str is not None:
        if parent:
            candidates.append(os.path.join(parent, date_str, plan_name))
        candidates.append(os.path.join("sessions", date_str, plan_name))
    candidates.append(os.path.join("sessions", plan_name))

    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Plan directory not found. Tried:\n  "
        + "\n  ".join(os.path.normpath(candidate) for candidate in candidates)
    )


def _unique_plan_dir(sessions_dir: str = "sessions", suffix: str = "") -> str:
    date_str = datetime.datetime.now().strftime("%y%m%d")
    dated_sessions_dir = os.path.join(sessions_dir, date_str)
    os.makedirs(dated_sessions_dir, exist_ok=True)

    suffix = str(suffix or "").strip("_")
    idx = 1
    while True:
        name = f"plan_{date_str}_{idx}"
        if suffix:
            name = f"{name}_{suffix}"
        candidate = os.path.join(dated_sessions_dir, name)
        if not os.path.exists(candidate):
            return os.path.abspath(candidate)
        idx += 1


def _ignore_branch_copy(_src, names):
    ignored = set()
    for name in names:
        if name in PLAN_BRANCH_SKIP_NAMES:
            ignored.add(name)
            continue
        if any(name.endswith(suffix) for suffix in PLAN_BRANCH_SKIP_SUFFIXES):
            ignored.add(name)
    return ignored


def _record_branch_metadata(branch_dir: str, source_dir: str) -> None:
    params_path = os.path.join(branch_dir, "planning_params.pkl")
    if not os.path.exists(params_path):
        return
    try:
        with open(params_path, "rb") as f:
            params = pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read branched planning params {params_path}: {exc}")
        return
    if not isinstance(params, dict):
        print(f"[WARN] Branched planning params is not a dict: {params_path}")
        return

    params["branched_from"] = {
        "source_plan_dir": os.path.abspath(source_dir),
        "branched_by": "scripts.desktop.execute_online --plan",
        "branched_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(params_path, "wb") as f:
        pickle.dump(params, f)


def _branch_plan_for_main(plan: str) -> str:
    source_dir = _resolve_plan_dir(plan)
    branch_dir = _unique_plan_dir("sessions", suffix=ONLINE_PLAN_SUFFIX)
    shutil.copytree(
        source_dir,
        branch_dir,
        ignore=_ignore_branch_copy,
        symlinks=True,
    )
    _record_branch_metadata(branch_dir, source_dir)
    print(f"[INFO] execute_online.py branched reference plan: {source_dir}")
    print(f"[INFO] execute_online.py execution branch: {branch_dir}")
    return branch_dir


def _split_geometry_entry(entry):
    geom, color = entry, None
    meta = {}
    if isinstance(entry, tuple):
        geom = entry[0]
        color = list(entry[1])[:3] if len(entry) > 1 else None
        for extra in entry[2:]:
            if isinstance(extra, dict):
                meta.update(extra)
            elif isinstance(extra, str):
                meta["style"] = extra
    return geom, color, meta


def _serialize_geometry_entry(entry):
    geom, color, meta = _split_geometry_entry(entry)

    if hasattr(geom, "vertices"):
        vertices = np.asarray(geom.vertices, dtype=np.float64)
        triangles = np.asarray(geom.triangles, dtype=np.int32)
        if vertices.size == 0 or not np.all(np.isfinite(vertices)):
            return None
        item = {
            "kind": "mesh",
            "vertices": vertices,
            "triangles": triangles,
            "color": color,
        }
        _copy_geometry_meta(item, meta)
        return item
    if hasattr(geom, "points"):
        points = np.asarray(geom.points, dtype=np.float64)
        if points.size == 0 or not np.all(np.isfinite(points)):
            return None
        if hasattr(geom, "lines"):
            lines = np.asarray(geom.lines, dtype=np.int32)
            if lines.size == 0:
                return None
            item = {
                "kind": "lineset",
                "points": points,
                "lines": lines,
                "color": color,
            }
            _copy_geometry_meta(item, meta)
            if hasattr(geom, "colors") and len(geom.colors) > 0:
                item["colors"] = np.asarray(geom.colors, dtype=np.float64)
            return item
        item = {
            "kind": "pcd",
            "points": points,
            "color": color,
        }
        _copy_geometry_meta(item, meta)
        return item
    return None


def _copy_geometry_meta(item, meta):
    name = meta.get("name")
    if name:
        item["name"] = str(name)
    transform = meta.get("transform")
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
            item["transform"] = transform
    style = meta.get("style")
    if style:
        item["style"] = str(style)
    if "alpha" in meta:
        try:
            alpha = float(meta["alpha"])
        except (TypeError, ValueError):
            alpha = None
        if alpha is not None and np.isfinite(alpha):
            item["alpha"] = float(np.clip(alpha, 0.0, 1.0))


def _serialize_geometries(geoms):
    out = []
    for entry in geoms:
        item = _serialize_geometry_entry(entry)
        if item is not None:
            out.append(item)
    return out


def _serialize_live_joint_state(payload):
    out = {}
    transforms = {}
    for name, transform in (payload.get("transforms") or {}).items():
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
            transforms[str(name)] = transform
    if transforms:
        out["transforms"] = transforms
    return out


def _put_latest(q, msg):
    while True:
        try:
            q.put_nowait(msg)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def _worker_main(
    plan_id,
    start_step,
    online_options,
    control_options,
    gui_q,
    display_q,
    live_q,
    cmd_q,
):
    from planning.execution import PlanningWorker

    worker = PlanningWorker(
        plan_id=plan_id,
        start_step=start_step,
        online_options=online_options,
        control_options=control_options,
        on_log=lambda msg: gui_q.put(("log", msg)),
        on_status=lambda st: gui_q.put(("status", st)),
        on_request_decision=lambda k, p: gui_q.put(("request_decision", k, p)),
        on_display_geometries=lambda g, t: _put_latest(
            display_q,
            ("display_geometries", _serialize_geometries(g), t),
        ),
        on_live_joint_state=lambda payload: _put_latest(
            live_q,
            ("live_joint_state", _serialize_live_joint_state(payload)),
        ),
        on_finished=lambda: gui_q.put(("finished",)),
        on_failed=lambda msg: gui_q.put(("failed", msg)),
    )

    def _command_loop():
        while True:
            cmd = cmd_q.get()
            if not cmd:
                continue
            kind = cmd[0]
            if kind == "decision":
                worker.on_decision(cmd[1])
            elif kind == "abort":
                worker.on_abort()
                break
            elif kind == "pick_adjust":
                worker.on_pick_grasp_adjust(cmd[1], cmd[2])
            elif kind == "pick_adjust_reset":
                worker.on_pick_grasp_adjust_reset()
            elif kind == "place_z_offset":
                worker.on_place_z_offset_changed(cmd[1])
            elif kind == "manual_place":
                worker.on_manual_place()
            elif kind == "subgoal_pcd_scan":
                worker.on_subgoal_pcd_scan()
            elif kind == "stone_pcd_mode":
                worker.on_stone_pcd_mode_changed(cmd[1])
            elif kind == "stone_mesh_mode":
                worker.on_stone_mesh_mode_changed(cmd[1])
            elif kind == "lidar_frame_mode":
                worker.on_lidar_frame_mode_changed(cmd[1])
            elif kind == "refresh":
                worker.on_refresh_requested()

    threading.Thread(target=_command_loop, daemon=True).start()
    worker.run()


def _start_gui_relay(window, gui_q, display_q, live_q):
    stop_event = threading.Event()

    def _relay():
        while not stop_event.is_set():
            try:
                msg = gui_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not msg:
                continue
            kind = msg[0]
            if kind == "log":
                window.append_log(msg[1])
            elif kind == "status":
                window.update_status(msg[1])
            elif kind == "request_decision":
                window.request_decision(msg[1], msg[2])
            elif kind == "finished":
                window.on_finished()
                stop_event.set()
                return
            elif kind == "failed":
                window.on_failed(msg[1])
                stop_event.set()
                return

    def _display_relay():
        while not stop_event.is_set():
            try:
                msg = display_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not msg:
                continue
            if msg[0] == "display_geometries":
                window.display_geometries(msg[1], msg[2])

    def _live_relay():
        while not stop_event.is_set():
            try:
                msg = live_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if not msg:
                continue
            if msg[0] == "live_joint_state":
                window.update_live_joint_state(msg[1])

    threads = [
        threading.Thread(target=_relay),
        threading.Thread(target=_display_relay),
        threading.Thread(target=_live_relay),
    ]
    for thread in threads:
        thread.start()
    return stop_event, threads


def _stop_gui_relay(stop_event, threads):
    stop_event.set()
    for thread in threads:
        thread.join(timeout=1.0)


def _close_queue(q):
    try:
        q.close()
        q.join_thread()
    except Exception:
        pass


def main():
    args = _parse_args()
    if args.plan is not None:
        plan_id = _branch_plan_for_main(args.plan)
        online_options = _build_online_options(args)
        online_options["seed_plan_dir"] = plan_id
        online_options["seed_plan_review"] = True
    else:
        plan_id = None
        online_options = _build_online_options(args)
    if args.live_joint_viewer:
        os.environ["STACKING_LIVE_JOINT_VIEWER"] = "1"
    else:
        os.environ["STACKING_LIVE_JOINT_VIEWER"] = "0"
    os.environ["STACKING_LIVE_JOINT_VIEWER_MIN_INTERVAL"] = str(
        max(0.1, args.live_joint_interval)
    )

    from gui import ExecutionWindow

    mp_ctx = mp.get_context("spawn")
    gui_q = mp_ctx.Queue()
    display_q = mp_ctx.Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
    live_q = mp_ctx.Queue(maxsize=LIVE_QUEUE_MAXSIZE)
    cmd_q = mp_ctx.Queue()

    window = ExecutionWindow()
    window.on_decide = lambda value: cmd_q.put(("decision", value))
    window.on_abort = lambda: cmd_q.put(("abort",))
    window.on_pick_adjust = lambda joint, delta: cmd_q.put(
        ("pick_adjust", joint, delta)
    )
    window.on_pick_adjust_reset = lambda: cmd_q.put(("pick_adjust_reset",))
    window.on_place_z_offset = lambda offset: cmd_q.put(
        ("place_z_offset", float(offset))
    )
    window.on_manual_place = lambda: cmd_q.put(("manual_place",))
    window.on_subgoal_pcd_scan = lambda: cmd_q.put(("subgoal_pcd_scan",))
    window.on_stone_pcd_mode = lambda enabled: cmd_q.put(
        ("stone_pcd_mode", bool(enabled))
    )
    window.on_stone_mesh_mode = lambda enabled: cmd_q.put(
        ("stone_mesh_mode", bool(enabled))
    )
    window.on_lidar_frame_mode = lambda enabled: cmd_q.put(
        ("lidar_frame_mode", bool(enabled))
    )
    window.on_refresh = lambda: cmd_q.put(("refresh",))

    worker = mp_ctx.Process(
        target=_worker_main,
        args=(
            plan_id,
            args.start_step,
            online_options,
            _build_control_options(args),
            gui_q,
            display_q,
            live_q,
            cmd_q,
        ),
    )
    worker.start()
    relay_stop, relay_threads = _start_gui_relay(window, gui_q, display_q, live_q)

    try:
        window.run()
    finally:
        _stop_gui_relay(relay_stop, relay_threads)
        try:
            cmd_q.put(("abort",))
        except Exception:
            pass

    worker.join(timeout=5.0)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=2.0)
    for q in (gui_q, display_q, live_q, cmd_q):
        _close_queue(q)


if __name__ == "__main__":
    main()
