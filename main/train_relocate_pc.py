from pathlib import Path

import torch.nn as nn
import wandb

from hand_env_utils.arg_utils import *
from hand_env_utils.teleop_env import create_relocate_env
from hand_env_utils.wandb_callback import WandbCallback
from stable_baselines3.common.torch_layers import PointNetExtractor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO


def setup_wandb(parser_config, exp_name):
    run = wandb.init(
        project="hand_teleop_ppo",
        name=exp_name,
        config=parser_config,
        monitor_gym=True,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=5)
    parser.add_argument('--bs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--object_name', type=str)

    args = parser.parse_args()
    object_name = args.object_name
    exp_keywords = ["ppopc", object_name, args.exp, str(args.seed)]
    env_iter = args.iter * 500 * args.n

    config = {
        'n_env_horizon': args.n,
        'object_name': args.object_name,
        'update_iteration': args.iter,
        'total_step': env_iter,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)

    wandb_run = setup_wandb(config, exp_name)


    def create_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=True)
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")

    print(env.observation_space, env.action_space)

    feature_extractor_class = PointNetExtractor
    feature_extractor_kwargs = {
        "pc_key": "relocate-point_cloud",
        # "feat_key": "relocate-point_cloud_feature",
        "local_channels": (64, 128, 256),
        "global_channels": (256,),
    }
    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.Tanh,
    }
    print("Initializing model")
    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * 500,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=args.seed,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(result_path / "log"),
                target_kl=0.02
                )
    print("Start learning")
    model.learn(
        total_timesteps=int(env_iter),
        callback=WandbCallback(
            model_save_freq=10,
            model_save_path=str(result_path / "model"),
        ),
    )
    wandb_run.finish()
