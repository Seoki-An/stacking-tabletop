#!/usr/bin/env python3
import os
from omegaconf import OmegaConf
import pickle
import copy
import numpy as np
import argparse

from planning.visualization import generate_path_with_opening_angle

from model import get_stone_model, get_excavator_model
from planning import (
    get_planner,
    normalize_joint_branches,
    regrasp_planning,
    trajectory_visualization_with_target,
    generate_path_with_opening_angle,
)

from utils import get_unique_dir

VISUALIZATION_ON = False


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--plan_path", type=str, required=True)
    args = argparser.parse_args()

    ## Parameters ##
    q_home = np.array(
        [np.pi / 2, np.pi / 4, -np.pi / 3, -np.pi / 2 + 0.2, 0.0, 0.0, 0.0, 0.0]
    )
    n_move = 40
    n_grasp = 20
    n_opening_angle = 10
    regrasp_xy_pos = np.array([2.0, 5.0])

    action_path = os.path.join(args.plan_path, "action_sequence.pkl")
    planning_params_path = os.path.join(args.plan_path, "planning_params.pkl")

    with open(planning_params_path, "rb") as f:
        planning_params = pickle.load(f)
    q_home = planning_params["q_home"]
    target_structure_offset = planning_params["target_structure_offset"]
    n_move = planning_params["n_move"]
    n_grasp = planning_params["n_grasp"]
    n_opening_angle = planning_params["n_opening_angle"]
    regrasp_xy_pos = planning_params["regrasp_xy_pos"]
    poses = planning_params.get("poses", planning_params.get("pose_data"))
    if poses is None:
        raise KeyError("planning_params must contain either 'poses' or 'pose_data'")

    video_dir = os.path.join(args.plan_path, "videos")
    video_dir = get_unique_dir(video_dir, prefix="motion")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    camera_center = [0, 0, 0]
    camera_position = [8.0, 5.0, 4.0]
    ################

    context, _ = get_planner()

    excavator_model, excavator_meshes = get_excavator_model()
    stone_meshes, stone_configs, _, _ = get_stone_model()

    with open(action_path, "rb") as f:
        action_sequence = pickle.load(f)

    pick_ids = {}
    scene_meshes = {}
    scene_configs = {}
    for id, mesh in stone_meshes.items():
        pos, quat = poses[id]
        stone_configs[id].pose.setPosition(pos)
        stone_configs[id].pose.setOrientation(quat)

        pose_T = stone_configs[id].pose.as_matrix()

        mesh = copy.deepcopy(mesh)
        mesh.transform(pose_T)
        scene_meshes[id] = mesh
        scene_configs[id] = stone_configs[id]
        pick_ids[id] = context.add_pick_body(stone_configs[id])

    motion_sequence = []
    for n_step, action in enumerate(action_sequence):
        # Motion planning
        place_pose = copy.deepcopy(action["pose"])
        place_pose[:2] += target_structure_offset[:2]
        target_id = action["stone_id"]

        pick_config = copy.deepcopy(stone_configs[target_id])

        place_config = copy.deepcopy(pick_config)
        place_config.pose.setPosition(place_pose[:3])
        place_config.pose.setOrientation(place_pose[3:])

        print("Start motion planning...")

        context.remove_body(pick_ids[target_id])
        result = regrasp_planning(
            context, pick_config, place_config, q_home, regrasp_xy_pos, n_move, n_grasp
        )
        context.add_place_body(place_config)
        path1 = result.q_path_sequence[0][:n_move]
        path2 = result.q_path_sequence[0][n_move:]
        path3 = result.q_path_sequence[1][:n_grasp]
        path4 = result.q_path_sequence[1][n_grasp:]
        path1, path2, path3, path4 = normalize_joint_branches(
            [path1, path2, path3, path4], q_home
        )
        motions = [path1, path2, path3, path4]
        if len(result.q_path_sequence) == 8:
            path1 = result.q_path_sequence[2][:n_move]
            path2 = result.q_path_sequence[2][n_move:]
            path3 = result.q_path_sequence[3][:n_grasp]
            path4 = result.q_path_sequence[3][n_grasp:]
            path1, path2, path3, path4 = normalize_joint_branches(
                [path1, path2, path3, path4], q_home
            )
            motions += [path1, path2, path3, path4]

        motion_sequence.append(motions)

        if VISUALIZATION_ON:
            save_path = os.path.join(video_dir, f"step_{n_step+1}.mp4")
            q_path, target_path = generate_path_with_opening_angle(
                result, n_opening_angle
            )
            scene_meshes.pop(target_id)
            trajectory_visualization_with_target(
                q_path,
                target_path,
                excavator_model,
                excavator_meshes,
                scene_meshes,
                stone_meshes[target_id],
                save_path,
                camera_center,
                camera_position,
            )

        target_mesh = copy.deepcopy(stone_meshes[target_id])
        target_mesh.transform(place_config.pose.as_matrix())
        scene_meshes[target_id] = target_mesh


if __name__ == "__main__":

    main()
