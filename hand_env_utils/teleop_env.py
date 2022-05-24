import numpy as np
from hand_teleop.env.rl_env.relocate_env import LabArmAllegroRelocateRLEnv
import os


def create_relocate_env(object_name, use_visual_obs, use_gui=False):
    if object_name == "mustard_bottle":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    elif object_name in ["tomato_soup_can", "potted_meat_can"]:
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    else:
        print(object_name)
        raise NotImplementedError
    rotation_reward_weight = 0
    env_params = dict(object_name=object_name, robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=use_gui)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = LabArmAllegroRelocateRLEnv(**env_params)

    if use_visual_obs:
        # Create camera
        camera_cfg = {
            "relocate_view": dict(position=np.array([-0.4, 0.4, 0.6]), look_at_dir=np.array([0.4, -0.4, -0.6]),
                                  right_dir=np.array([-1, -1, 0]), fov=np.deg2rad(69.4), resolution=(64, 64)),
        }
        env.setup_camera_from_config(camera_cfg)

        # Specify modality
        empty_info = {}  # level empty dict for now, reserved for future
        camera_info = {"relocate_view": {"point_cloud": empty_info}}
        env.setup_visual_obs_config(camera_info)
        return env

    return env
