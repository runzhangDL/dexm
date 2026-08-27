"""
Vectorized Dexm Environment Interface for Reinforcement Learning (PPO).

Bridges DexmBuilder with PyTorch tensors for Actor and Critic networks.
- Actor receives: RGB Camera Tensor (N, 3, H, W) and Proprioceptive Joint Observations (N, 18).
- Critic receives: Asymmetric Privileged State Vector (N, 151) comprising:
    - Robot Arms: 54 dims
    - Cube / Box: 13 dims
    - Deformable Cable Keypoints (16 nodes): 48 dims
    - Precomputed Geometric Relational Offsets: 24 dims
    - Contact & Normal Forces: 8 dims
    - One-hot Stage ID: 4 dims
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import warp as wp
from dexm.builder import DexmBuilder


class DexmVecEnv:
    """
    Vectorized environment wrapping DexmBuilder for parallel simulation and PPO training.
    """

    def __init__(
        self,
        num_envs: int = 8,
        max_episode_steps: int = 200,
        action_scale: float = 0.05,
        max_joint_vel: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_graph: bool = True,
        table_z: float = 0.0,
        **builder_kwargs,
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.max_joint_vel = max_joint_vel
        self.device = torch.device(device)
        self.table_z = table_z

        # Initialize underlying vectorized Newton builder
        self.builder = DexmBuilder(
            worlds_count=self.num_envs,
            use_graph=use_graph,
            enable_camera=True,
            **builder_kwargs,
        )

        self.model = self.builder.model
        self.fps = self.builder.fps
        self.dt = self.builder.frame_dt

        # Track simulation steps & stages per environment
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_stage = torch.ones(self.num_envs, dtype=torch.long, device=self.device)  # 1 to 4

        # Number of cable keypoint nodes to subsample (16 keypoints -> 48 dims)
        self.num_cable_nodes = 16
        cable_bodies_per_env = len(self.builder.cable_bodies) // self.num_envs
        self.cable_subsample_idx = np.linspace(
            0, cable_bodies_per_env - 1, self.num_cable_nodes, dtype=int
        )

        # Joint limit tensors for clamping (18 DoF per env: 9 arm1 + 9 arm2)
        joint_lower_np = self.model.joint_limit_lower.numpy()
        joint_upper_np = self.model.joint_limit_upper.numpy()

        self.q_min = torch.tensor(
            [joint_lower_np[i] if not np.isneginf(joint_lower_np[i]) else -2.89 for i in range(18)],
            dtype=torch.float32,
            device=self.device,
        )
        self.q_max = torch.tensor(
            [joint_upper_np[i] if not np.isposinf(joint_upper_np[i]) else 2.89 for i in range(18)],
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reset environments and return initial observations.
        Returns:
            images: (num_envs, 3, H, W) float32 tensor
            actor_obs: (num_envs, 18) float32 proprioceptive joint tensor
            critic_state: (num_envs, 151) float32 privileged state tensor
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            self.builder.reset()
            self.episode_steps.zero_()
            self.current_stage.fill_(1)
        else:
            self.episode_steps[env_ids] = 0
            self.current_stage[env_ids] = 1

        images, actor_obs = self.get_actor_obs()
        critic_state = self.get_critic_state()
        return images, actor_obs, critic_state

    def get_actor_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract observation inputs for the Actor network:
        1. RGB Camera Images: (num_envs, 3, H, W)
        2. Robot Proprioceptive Joint Positions: (num_envs, 18)
        """
        if self.builder.enable_camera:
            self.model.bvh_refit_shapes(self.builder.state_0)
            self.builder.camera_sensor.update(
                state=self.builder.state_0,
                camera_transforms=self.builder.camera_transforms,
                camera_rays=self.builder.camera_rays,
                color_image=self.builder._camera_color_output,
                clear_data=self.builder._camera_clear_data,
            )
            rgba_np = self.builder.camera_sensor.utils.to_rgba_from_color(
                self.builder._camera_color_output
            ).numpy()
            rgb_tensor = torch.from_numpy(rgba_np[:, :, :, :3]).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        else:
            rgb_tensor = torch.zeros(
                (self.num_envs, 3, self.builder.camera_height, self.builder.camera_width),
                device=self.device,
                dtype=torch.float32,
            )

        joint_q_np = self.builder.state_0.joint_q.numpy()
        dofs_per_env = self.model.joint_dof_count // self.num_envs
        joint_obs = []
        for e in range(self.num_envs):
            start = e * dofs_per_env
            joint_obs.append(joint_q_np[start : start + 18])
        joint_obs_tensor = torch.tensor(np.array(joint_obs), dtype=torch.float32, device=self.device)

        return rgb_tensor, joint_obs_tensor

    def get_critic_state(self) -> torch.Tensor:
        """
        Extract the 151-dimensional privileged state vector for the Critic network:
        A. Robot Arms (54 dims)
        B. Cube / Box (13 dims)
        C. Deformable Cable (48 dims)
        D. Precomputed Relational Offsets (24 dims)
        E. Contact & Normal Forces (8 dims)
        F. Current Stage ID (4 dims)
        Total: 151 dims
        """
        body_q_np = self.builder.state_0.body_q.numpy()
        body_qd_np = self.builder.state_0.body_qd.numpy()
        joint_q_np = self.builder.state_0.joint_q.numpy()
        joint_qd_np = self.builder.state_0.joint_qd.numpy()

        bodies_per_env = self.model.body_count // self.num_envs
        dofs_per_env = self.model.joint_dof_count // self.num_envs
        cable_per_env = len(self.builder.cable_bodies) // self.num_envs

        critic_states = []

        for e in range(self.num_envs):
            body_offset = e * bodies_per_env
            dof_offset = e * dofs_per_env
            cable_offset = e * cable_per_env

            # A. Robot Arms (54 dims)
            q_arm = joint_q_np[dof_offset : dof_offset + 18]
            qd_arm = joint_qd_np[dof_offset : dof_offset + 18]

            hand_r_idx = body_offset + 10
            hand_l_idx = body_offset + 24
            ee_r_pose = body_q_np[hand_r_idx]
            ee_l_pose = body_q_np[hand_l_idx]

            grip_r_width = q_arm[7] + q_arm[8]
            grip_l_width = q_arm[16] + q_arm[17]
            grip_r_vel = qd_arm[7] + qd_arm[8]
            grip_l_vel = qd_arm[16] + qd_arm[17]
            gripper_features = np.array([grip_l_width, grip_r_width, grip_l_vel, grip_r_vel], dtype=np.float32)

            arms_features = np.concatenate([q_arm, qd_arm, ee_l_pose, ee_r_pose, gripper_features])

            # B. Cube / Box (13 dims)
            cube_idx = body_offset + bodies_per_env - 1
            cube_pose = body_q_np[cube_idx]
            cube_vel = body_qd_np[cube_idx]
            cube_features = np.concatenate([cube_pose, cube_vel])

            # C. Deformable Cable (48 dims)
            cable_start = self.builder.cable_bodies[cable_offset]
            cable_positions = []
            for node_idx in self.cable_subsample_idx:
                body_id = cable_start + node_idx
                cable_positions.append(body_q_np[body_id][:3])
            cable_features = np.concatenate(cable_positions)

            # D. Precomputed Geometric Relational Offsets (24 dims)
            p_ee_l = ee_l_pose[:3]
            p_ee_r = ee_r_pose[:3]
            p_cube = cube_pose[:3]
            p_tip_a = cable_positions[0]
            p_tip_b = cable_positions[-1]
            p_cable_mid = cable_positions[len(cable_positions) // 2]

            v_ee_l_to_cube = p_cube - p_ee_l
            v_ee_r_to_cube = p_cube - p_ee_r
            v_ee_l_to_tip_a = p_tip_a - p_ee_l
            v_ee_r_to_tip_b = p_tip_b - p_ee_r
            v_cube_to_cable_mid = p_cable_mid - p_cube
            dist_ee_l_to_ee_r = np.array([np.linalg.norm(p_ee_l - p_ee_r)], dtype=np.float32)

            height_diffs = np.array([
                p_cube[2] - self.table_z,
                p_tip_a[2] - p_cube[2],
                p_tip_b[2] - p_cube[2],
                p_cable_mid[2] - p_cube[2],
                p_ee_l[2] - p_cube[2],
                p_ee_r[2] - p_cube[2],
                p_ee_l[2] - self.table_z,
                p_ee_r[2] - self.table_z,
            ], dtype=np.float32)

            rel_offsets = np.concatenate([
                v_ee_l_to_cube,
                v_ee_r_to_cube,
                v_ee_l_to_tip_a,
                v_ee_r_to_tip_b,
                v_cube_to_cable_mid,
                dist_ee_l_to_ee_r,
                height_diffs,
            ])

            # E. Contact & Normal Forces (8 dims)
            contact_features = np.zeros(8, dtype=np.float32)

            # F. Current Stage ID (4 dims)
            stage_val = int(self.current_stage[e].item())
            stage_one_hot = np.zeros(4, dtype=np.float32)
            stage_one_hot[min(stage_val - 1, 3)] = 1.0

            full_state = np.concatenate([
                arms_features,
                cube_features,
                cable_features,
                rel_offsets,
                contact_features,
                stage_one_hot,
            ])
            critic_states.append(full_state)

        return torch.tensor(np.array(critic_states), dtype=torch.float32, device=self.device)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one policy step across all parallel environments.

        Action Post-Processing Pipeline:
        1. Clamp / Squash to [-1.0, 1.0]
        2. Scale by dt * max_joint_velocity (action_scale)
        3. Compute Delta: q_target = q_current + delta_q
        4. Hard-Clip to Physical Hardware Limits: [q_min, q_max]
        5. Pass to Newton Solver PD controller targets
        """
        # Step 1: Clamp / Squash raw action
        action_clamped = torch.clamp(action, -1.0, 1.0)

        # Step 2: Scale delta joint positions
        delta_q = action_clamped * self.action_scale

        # Step 3: Compute current + delta
        target_q_full = self.model.joint_target_q.numpy()
        dofs_per_env = self.model.joint_dof_count // self.num_envs

        for e in range(self.num_envs):
            start = e * dofs_per_env
            curr_q_e = torch.tensor(target_q_full[start : start + 18], dtype=torch.float32, device=self.device)
            target_q_e = curr_q_e + delta_q[e]

            # Step 4: Hard-Clip to physical limits
            target_q_e = torch.clamp(target_q_e, self.q_min, self.q_max)

            # Step 5: Assign into model target buffer
            target_q_full[start : start + 18] = target_q_e.cpu().numpy()

        self.model.joint_target_q.assign(target_q_full)

        # Advance Newton physics simulation
        self.builder.step()
        self.episode_steps += 1

        next_images, next_actor_obs = self.get_actor_obs()
        next_critic_state = self.get_critic_state()

        rewards = self._compute_reward(next_critic_state, action_clamped)
        dones = (self.episode_steps >= self.max_episode_steps).unsqueeze(1).float()

        done_indices = torch.where(dones.squeeze(1) > 0.5)[0]
        if len(done_indices) > 0:
            self.episode_steps[done_indices] = 0

        infos = {"stage": self.current_stage.clone()}
        return (next_images, next_actor_obs), next_critic_state, rewards, dones, infos

    def _compute_reward(self, critic_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Placeholder Reward Function for Knot-Tying on a Present Box.
        """
        v_l_tip = critic_state[:, 115:118]
        v_r_tip = critic_state[:, 118:121]
        dist_l = torch.norm(v_l_tip, dim=-1, keepdim=True)
        dist_r = torch.norm(v_r_tip, dim=-1, keepdim=True)

        action_penalty = torch.sum(action ** 2, dim=-1, keepdim=True) * 0.01
        reward = -(dist_l + dist_r) - action_penalty
        return reward
