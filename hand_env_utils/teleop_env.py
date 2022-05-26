import os

from hand_teleop.env.rl_env.relocate_env import LabArmAllegroRelocateRLEnv
from hand_teleop.real_world import task_setting


def create_relocate_env(object_name, use_visual_obs, use_gui=False):
    if object_name == "mustard_bottle":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    elif object_name in ["tomato_soup_can", "potted_meat_can"]:
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    else:
        print(object_name)
        raise NotImplementedError
    rotation_reward_weight = 1
    env_params = dict(object_name=object_name, robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=True)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = LabArmAllegroRelocateRLEnv(**env_params)

    if use_visual_obs:
        # Create camera and setup visual modality
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])

        return env

    return env