"""
PPO (Proximal Policy Optimization) Actor-Critic Trainer for Bimanual Manipulation.

Implements Algorithm 1 (Actor-Critic PPO with Asymmetric Information):
- Actor: Visual-proprioceptive policy (RGB camera + 18-DoF robot joint state)
- Critic: Privileged state baseline (151-dim multi-object physical & relational state)
"""

import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from dexm.model import Actor, Critic
from dexm.env import DexmVecEnv
from dexm.buffer import RolloutBuffer


class PPOTrainer:
    """
    PPO Actor-Critic Trainer orchestrating rollout collection, GAE computation,
    and minibatch surrogate optimization.
    """

    def __init__(
        self,
        env: DexmVecEnv,
        actor: Actor,
        critic: Critic,
        # Rollout & Batching Hyperparameters
        num_steps: int = 64,             # Timesteps per iteration (T)
        num_minibatches: int = 4,        # Number of minibatches per epoch
        update_epochs: int = 4,          # Optimization epochs per iteration (K)
        # Optimization Hyperparameters
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_coef: float = 0.2,          # PPO epsilon clip range
        vf_coef: float = 0.5,            # Value loss coefficient
        ent_coef: float = 0.01,          # Entropy bonus coefficient
        max_grad_norm: float = 0.5,      # Gradient clipping threshold
        gamma: float = 0.99,             # Discount factor
        gae_lambda: float = 0.95,        # GAE lambda parameter
        # Checkpointing
        save_dir: str = "_static/checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.num_steps = num_steps
        self.num_envs = env.num_envs
        self.total_samples = self.num_steps * self.num_envs
        self.minibatch_size = self.total_samples // num_minibatches
        self.update_epochs = update_epochs

        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup Optimizers
        # Note: actor.dinov3_backbone is frozen (requires_grad=False)
        actor_trainable_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = optim.AdamW(actor_trainable_params, lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        # Initialize Rollout Buffer
        image_shape = (3, self.env.builder.camera_height, self.env.builder.camera_width)
        self.buffer = RolloutBuffer(
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            image_shape=image_shape,
            obs_dim=self.actor.obs_dim,
            critic_dim=151,
            action_dim=self.actor.action_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

        # Storage for global tracking
        self.global_step = 0
        self.iteration = 0

    def collect_rollouts(
        self,
        curr_images: torch.Tensor,
        curr_obs: torch.Tensor,
        curr_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Run current policy π_θ in parallel environments for T timesteps.
        Computes GAE advantage estimates and fills the rollout buffer.
        """
        self.actor.eval()
        self.critic.eval()
        self.buffer.clear()

        rollout_rewards = []

        for step in range(self.num_steps):
            with torch.no_grad():
                # 1. Sample action from Actor policy
                actions, log_probs, _ = self.actor.get_action(curr_images, curr_obs)
                # 2. Estimate state value from Critic baseline
                values = self.critic(curr_state)

            # 3. Step vectorized environment
            (next_images, next_obs), next_state, rewards, dones, infos = self.env.step(actions)

            # 4. Record transition into buffer
            self.buffer.insert(
                image=curr_images,
                obs=curr_obs,
                critic_state=curr_state,
                action=actions,
                log_prob=log_probs,
                reward=rewards,
                done=dones,
                value=values,
            )

            rollout_rewards.append(rewards.mean().item())
            curr_images, curr_obs, curr_state = next_images, next_obs, next_state
            self.global_step += self.num_envs

        # Compute bootstrap value for final timestep
        with torch.no_grad():
            last_value = self.critic(curr_state)
            last_done = dones

        # 5. Compute Generalized Advantage Estimates (GAE)
        self.buffer.compute_returns_and_advantages(last_value, last_done)

        metrics = {
            "mean_step_reward": float(np.mean(rollout_rewards)),
            "sum_step_reward": float(np.sum(rollout_rewards)),
        }
        return curr_images, curr_obs, curr_state, metrics

    def update_policy(self) -> Dict[str, float]:
        """
        Optimize surrogate loss L wrt θ and value loss wrt φ with K epochs
        and minibatches of size M <= N * T.
        """
        self.actor.train()
        self.critic.train()

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []

        for epoch in range(self.update_epochs):
            for batch in self.buffer.get_generator(self.minibatch_size):
                b_images = batch["images"]
                b_obs = batch["obs"]
                b_states = batch["critic_states"]
                b_actions = batch["actions"]
                b_old_log_probs = batch["log_probs"]
                b_advantages = batch["advantages"]
                b_returns = batch["returns"]

                # 1. Forward Actor with previous actions
                _, new_log_probs, entropy = self.actor.get_action(b_images, b_obs, b_actions)

                # 2. Probability ratio r_t(θ)
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                # Diagnostic KL & Clipping metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    clip_fractions.append(clip_frac)

                # 3. Clipped Surrogate Policy Loss
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 4. Value Function Loss
                new_values = self.critic(b_states)
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # 5. Entropy Bonus
                entropy_loss = -entropy.mean()

                # Total Loss
                total_actor_loss = policy_loss + self.ent_coef * entropy_loss
                total_critic_loss = self.vf_coef * value_loss

                # Backprop & Step Actor
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Backprop & Step Critic
                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }

    def train(self, total_iterations: int = 100, save_freq: int = 10):
        """
        Execute full training loop for Algorithm 1 (PPO Actor-Critic).
        """
        print(f"=== Starting PPO Training ===")
        print(f"Num Envs: {self.num_envs} | Rollout Steps: {self.num_steps} | Total Samples/Iter: {self.total_samples}")
        print(f"Minibatch Size: {self.minibatch_size} | Update Epochs: {self.update_epochs} | Total Iterations: {total_iterations}")

        # Reset environment
        curr_images, curr_obs, curr_state = self.env.reset()

        start_time = time.time()

        for iteration in range(1, total_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            # 1. Collect Rollout
            curr_images, curr_obs, curr_state, rollout_metrics = self.collect_rollouts(
                curr_images, curr_obs, curr_state
            )

            # 2. Optimize Policy & Value baseline
            update_metrics = self.update_policy()

            iter_duration = time.time() - iter_start
            fps = self.total_samples / max(iter_duration, 1e-4)

            # Logging
            print(
                f"[Iter {iteration:4d}/{total_iterations}] "
                f"Reward: {rollout_metrics['mean_step_reward']:+7.3f} | "
                f"Loss(P): {update_metrics['policy_loss']:+7.4f} | "
                f"Loss(V): {update_metrics['value_loss']:7.4f} | "
                f"Ent: {update_metrics['entropy']:6.3f} | "
                f"KL: {update_metrics['approx_kl']:.4f} | "
                f"FPS: {fps:5.1f}"
            )

            # Save Checkpoints
            if iteration % save_freq == 0 or iteration == total_iterations:
                self.save_checkpoint(iteration)

        total_time = time.time() - start_time
        print(f"=== Training Complete in {total_time:.2f}s ===")

    def save_checkpoint(self, iteration: int):
        """Save Actor and Critic weights to checkpoint file."""
        ckpt_path = self.save_dir / f"ppo_dexm_iter_{iteration:04d}.pt"
        torch.save(
            {
                "iteration": iteration,
                "global_step": self.global_step,
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            str(ckpt_path),
        )
        print(f"  -> Checkpoint saved to: {ckpt_path}")
