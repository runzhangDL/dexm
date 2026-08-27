"""
Rollout Storage Buffer with Generalized Advantage Estimation (GAE) for PPO.

Stores trajectories from parallel environments and generates minibatches for
Actor-Critic optimization.
"""

from typing import Generator, Dict, Any, Tuple
import torch


class RolloutBuffer:
    """
    Rollout buffer storing transitions across T timesteps for N vectorized environments.
    Computes GAE advantages and generates minibatches for PPO updates.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        image_shape: Tuple[int, int, int],  # (3, H, W)
        obs_dim: int = 18,
        critic_dim: int = 151,
        action_dim: int = 18,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.image_shape = image_shape
        self.obs_dim = obs_dim
        self.critic_dim = critic_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.total_samples = num_steps * num_envs

        # Allocate memory buffers
        self.images = torch.zeros((num_steps, num_envs, *image_shape), dtype=torch.float32, device=self.device)
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=self.device)
        self.critic_states = torch.zeros((num_steps, num_envs, critic_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=self.device)

        self.advantages = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((num_steps, num_envs, 1), dtype=torch.float32, device=self.device)

        self.step = 0

    def insert(
        self,
        image: torch.Tensor,
        obs: torch.Tensor,
        critic_state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        """Insert a single timestep transition across all N environments."""
        if self.step >= self.num_steps:
            raise IndexError(f"RolloutBuffer overflow: step {self.step} >= num_steps {self.num_steps}")

        self.images[self.step].copy_(image)
        self.obs[self.step].copy_(obs)
        self.critic_states[self.step].copy_(critic_state)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)

        self.step += 1

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, last_done: torch.Tensor
    ):
        """
        Compute Generalized Advantage Estimation (GAE) and discounted returns.
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
        """
        last_gae = torch.zeros_like(last_value)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_generator(
        self, minibatch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Flatten buffer across (num_steps, num_envs) and yield random minibatches for PPO updates.
        """
        # Flatten tensors
        b_images = self.images.view(-1, *self.image_shape)
        b_obs = self.obs.view(-1, self.obs_dim)
        b_critic_states = self.critic_states.view(-1, self.critic_dim)
        b_actions = self.actions.view(-1, self.action_dim)
        b_log_probs = self.log_probs.view(-1, 1)
        b_advantages = self.advantages.view(-1, 1)
        b_returns = self.returns.view(-1, 1)
        b_values = self.values.view(-1, 1)

        # Normalize advantages across the whole batch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = self.total_samples
        indices = torch.randperm(total_samples, device=self.device)

        for start in range(0, total_samples, minibatch_size):
            end = min(start + minibatch_size, total_samples)
            batch_idx = indices[start:end]

            yield {
                "images": b_images[batch_idx],
                "obs": b_obs[batch_idx],
                "critic_states": b_critic_states[batch_idx],
                "actions": b_actions[batch_idx],
                "log_probs": b_log_probs[batch_idx],
                "advantages": b_advantages[batch_idx],
                "returns": b_returns[batch_idx],
                "values": b_values[batch_idx],
            }

    def clear(self):
        """Reset step pointer for next rollout iteration."""
        self.step = 0
