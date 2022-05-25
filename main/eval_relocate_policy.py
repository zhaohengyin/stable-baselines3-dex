import numpy as np

from hand_env_utils.teleop_env import create_relocate_env
from stable_baselines3.dapg import DAPG
from stable_baselines3.ppo import PPO

if __name__ == '__main__':
    checkpoint_path = "results/ppo-mustard_bottle-pc_bs_1000-100/model/model_254.zip"
    use_visual_obs = True
    object_name = checkpoint_path.split("/")[1].split("-")[1]
    algorithm_name = checkpoint_path.split("/")[1].split("-")[0]
    env = create_relocate_env(object_name, use_visual_obs=False, use_gui=True)

    device = "cuda:0"
    if algorithm_name == "ppo":
        policy = PPO.load(checkpoint_path, env, device)
    elif algorithm_name == "dapg":
        policy = DAPG.load(checkpoint_path, env, device)
    else:
        raise NotImplementedError

    viewer = env.render(mode="human")

    done = False
    manual_action = False
    action = np.zeros(22)
    while not viewer.closed:
        reward_sum = 0
        obs = env.reset()
        for i in range(250):
            if manual_action:
                action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
            else:
                action = policy.predict(observation=obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            if env.viewer.window.key_down("enter"):
                manual_action = True
            elif env.viewer.window.key_down("p"):
                manual_action = False

        print(f"Reward: {reward_sum}")
