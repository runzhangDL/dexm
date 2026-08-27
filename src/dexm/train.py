"""
PPO Training Entry Point for Dexm Bimanual Manipulation.

Usage:
    python src/dexm/train.py --num_envs 4 --num_steps 32 --iterations 100
"""

import argparse
import torch

from dexm.model import Actor, Critic
from dexm.env import DexmVecEnv
from dexm.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for Bimanual Manipulation in Dexm")
    # Environment args
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel vectorized simulation environments")
    parser.add_argument("--num_steps", type=int, default=32, help="Rollout timesteps per iteration (T)")
    parser.add_argument("--max_episode_steps", type=int, default=150, help="Max steps before episode truncation")
    parser.add_argument("--action_scale", type=float, default=0.05, help="Action delta joint angle scale (rad/step)")
    # PPO args
    parser.add_argument("--iterations", type=int, default=50, help="Total PPO training iterations")
    parser.add_argument("--update_epochs", type=int, default=4, help="PPO optimization epochs per iteration (K)")
    parser.add_argument("--num_minibatches", type=int, default=2, help="Number of minibatches per epoch")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate for Actor MLP")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for Critic MLP")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO surrogate clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    # Model args
    parser.add_argument("--weights_path", type=str, default="/home/run/Downloads/dinov3_vit7b16.pth", help="DINOv3 backbone weights path")
    parser.add_argument("--save_dir", type=str, default="_static/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--save_freq", type=int, default=10, help="Checkpoint saving frequency (iterations)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Instantiate Vectorized Environment
    print("Building Vectorized Environment...")
    env = DexmVecEnv(
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        action_scale=args.action_scale,
        device=device,
    )

    # 2. Instantiate Actor and Critic
    print("Initializing Actor and Critic models...")
    actor = Actor(
        action_dim=18,
        obs_dim=18,
        weights_path=args.weights_path,
    ).to(device)

    critic = Critic(
        critic_dim=151,
    ).to(device)

    # 3. Instantiate PPO Trainer
    trainer = PPOTrainer(
        env=env,
        actor=actor,
        critic=critic,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        clip_coef=args.clip_coef,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        save_dir=args.save_dir,
        device=device,
    )

    # 4. Run Training
    trainer.train(total_iterations=args.iterations, save_freq=args.save_freq)


if __name__ == "__main__":
    main()
