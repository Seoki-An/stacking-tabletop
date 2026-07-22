#!/usr/bin/env python3
import datetime
import os
import pickle
import rclpy
from ros2 import PoseArraySubscriber, PhasePublisher

if __name__ == "__main__":

    rclpy.init()
    phase_pub = PhasePublisher()
    pose_array_sub = PoseArraySubscriber()

    print("Waiting for estimated poses...")
    while not pose_array_sub.get_flag():
        rclpy.spin_once(pose_array_sub, timeout_sec=0.01)
    print("Received estimated poses")
    pose_array_sub.reset_get_flag()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sessions_dir = "sessions"
    os.makedirs(sessions_dir, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%y%m%d")
    counter = 1
    while True:
        filename = f"pose_data_{date_str}_{counter}.pkl"
        save_path = os.path.join(sessions_dir, filename)
        if not os.path.exists(save_path):
            break
        counter += 1

    with open(save_path, "wb") as f:
        pickle.dump(pose_array_sub.pose_id, f)

    rel_path = os.path.relpath(save_path)
    print(f"Saving field poses complete - save path: {save_path}")
    print(f'Update config.yml: environment.action.pose_data_path: "{rel_path}"')
