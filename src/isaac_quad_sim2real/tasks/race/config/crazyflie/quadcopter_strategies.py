# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations
from torch._tensor import Tensor

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            # Add additional reward components that are not in env.rew
            keys.extend(['gate_passed'])  # Add gate_passed reward component
            # Add time reward component
            keys.extend(['time'])
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }
        
        # Initialize previous progress for reward calculation
        self._prev_progress_global = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Initialize gate passing state variables (maintained in strategy, not env)
        # These are used for gate passing detection and tracking
        self._prev_drone_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._idx_wp = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._n_gates_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Initialize parameters (will be randomized during training if domain randomization is enabled)
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def _check_gate_passing(self, prev_pos: torch.Tensor, curr_pos: torch.Tensor) -> torch.Tensor:
        """
        Check if the line segment from prev_pos to curr_pos passes through the current gate.
        
        Args:
            prev_pos: Previous drone position (num_envs, 3)
            curr_pos: Current drone position (num_envs, 3)
            
        Returns:
            gate_passed: Boolean tensor indicating which environments passed their gate (num_envs,)
        """
        num_envs = prev_pos.shape[0]
        gate_passed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # Get current gate info for all environments (use strategy's state)
        current_gate_idx = self._idx_wp  # shape: (num_envs,)
        gate_pos = self.env._waypoints[current_gate_idx, :3]  # shape: (num_envs, 3)
        gate_normal = self.env._normal_vectors[current_gate_idx, :]  # shape: (num_envs, 3)
        
        # Line segment direction
        segment_dir = curr_pos - prev_pos  # shape: (num_envs, 3)
        segment_length = torch.linalg.norm(segment_dir, dim=1, keepdim=True)  # shape: (num_envs, 1)
        
        # Avoid division by zero for stationary drones
        valid_movement = segment_length.squeeze(1) > 1e-6  # shape: (num_envs,)
        
        # Normalize segment direction
        segment_dir_norm = segment_dir / (segment_length + 1e-8)  # shape: (num_envs, 3)
        
        # Check if segment intersects the gate plane
        # Plane equation: (P - gate_pos) · gate_normal = 0
        # Line equation: P = prev_pos + t * segment_dir, where t ∈ [0, 1]
        # Solve for t: (prev_pos + t * segment_dir - gate_pos) · gate_normal = 0
        # t = (gate_pos - prev_pos) · gate_normal / (segment_dir · gate_normal)
        
        numerator = torch.sum((gate_pos - prev_pos) * gate_normal, dim=1)  # shape: (num_envs,)
        denominator = torch.sum(segment_dir * gate_normal, dim=1)  # shape: (num_envs,)
        
        # Avoid division by zero (segment parallel to plane)
        valid_intersection = torch.abs(denominator) > 1e-6  # shape: (num_envs,)
        
        # Calculate t parameter
        t = numerator / (denominator + 1e-8)  # shape: (num_envs,)
        
        # Check if intersection is within the segment (0 <= t <= 1)
        valid_t = (t >= 0) & (t <= 1)  # shape: (num_envs,)
        
        # Calculate intersection point
        intersection_point = prev_pos + t.unsqueeze(1) * segment_dir  # shape: (num_envs, 3)
        
        # Check if intersection point is within the gate rectangle
        # Transform intersection point to gate frame
        intersection_in_gate_frame, _ = subtract_frame_transforms(
            gate_pos,
            self.env._waypoints_quat[current_gate_idx, :],
            intersection_point
        )  # shape: (num_envs, 3)
        
        # Gate is a square with side length gate_side, centered at origin in gate frame
        # X should be ~0 (on the plane), Y and Z should be within [-gate_side/2, gate_side/2]
        # Get gate_side from config (could be attribute or dict key)
        if isinstance(self.env._gate_model_cfg_data, dict):
            gate_side = self.env._gate_model_cfg_data.get('gate_side', 1.0)
        elif hasattr(self.env._gate_model_cfg_data, 'gate_side'):
            gate_side = getattr(self.env._gate_model_cfg_data, 'gate_side', 1.0)
        else:
            gate_side = 1.0
        gate_half_side = gate_side / 2.0
        within_y = torch.abs(intersection_in_gate_frame[:, 1]) <= gate_half_side  # shape: (num_envs,)
        within_z = torch.abs(intersection_in_gate_frame[:, 2]) <= gate_half_side  # shape: (num_envs,)
        within_gate = within_y & within_z  # shape: (num_envs,)
        
        # Check direction: should pass through gate in the correct direction
        # Note: gate_normal points opposite to flight direction (arrow shows -gate_normal)
        # So we need segment_dir · gate_normal < 0 for correct direction
        correct_direction = denominator < 0  # shape: (num_envs,)
        
        # Combine all conditions
        gate_passed = valid_movement & valid_intersection & valid_t & within_gate & correct_direction
        
        return gate_passed

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # Get current drone position
        curr_drone_pos_w = self.env._robot.data.root_link_pos_w  # shape: (num_envs, 3)
        
        # Check gate passing using line segment intersection with gate plane (use strategy's method and state)
        gate_passed = self._check_gate_passing(
            self._prev_drone_pos_w,  # Previous position (from strategy)
            curr_drone_pos_w          # Current position
        )  # shape: (num_envs,)
        
        # Update previous position for next timestep (in strategy)
        self._prev_drone_pos_w = curr_drone_pos_w.clone()
        
        ids_gate_passed = torch.where(gate_passed)[0]
        
        # update gate index and count gates passed (in strategy)
        self._idx_wp[ids_gate_passed] = (self._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        self._n_gates_passed[ids_gate_passed] += 1

        # set desired positions in the world frame (in strategy)
        self._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self._idx_wp[ids_gate_passed], :2]
        self._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self._idx_wp[ids_gate_passed], 2]
        
        # Sync to env if it exists (for backward compatibility)
        if hasattr(self.env, '_idx_wp'):
            self.env._idx_wp[ids_gate_passed] = self._idx_wp[ids_gate_passed]
        if hasattr(self.env, '_n_gates_passed'):
            self.env._n_gates_passed[ids_gate_passed] = self._n_gates_passed[ids_gate_passed]
        if hasattr(self.env, '_desired_pos_w'):
            self.env._desired_pos_w[ids_gate_passed] = self._desired_pos_w[ids_gate_passed]

        # calculate progress via distance to goal
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1)
        distance_to_goal = torch.tanh(distance_to_goal/3.0)
        progress = 1 - distance_to_goal  # distance_to_goal is between 0 and 1 where 0 means the drone reached the goal
        
        # calculate global progress: normal progress + number of gates passed
        progress_global = progress + self._n_gates_passed.float()
        
        # calculate progress difference for reward (always calculate, used in both train and test)
        progress_diff = progress_global - self._prev_progress_global
        self._prev_progress_global = progress_global.clone()
        
        # Store progress_diff for external access (e.g., in play_race.py)
        self.env._current_progress_diff = progress_diff

        # compute crashed environments if contact detected for 100 timesteps
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        # 计算奖励 - 训练和测试使用相同的奖励函数
        # 首先创建不带scale的原始奖励字典
        raw_rewards = {
            "progress_goal": progress_diff,  # 不带scale的原始progress_diff
            "gate_passed": gate_passed.float(),  # 不带scale的原始gate_passed (0或1)
            "crash": crashed,  # 不带scale的原始crash (0或1)
            "time": torch.ones(self.num_envs, device=self.device),  # 不带scale的原始time (1.0)
        }
        
        # 将原始奖励信息存储到环境中
        self.env._raw_rewards = raw_rewards
        
        # 训练时计算带scale的奖励
        if self.cfg.is_train:
            # 应用奖励参数得到最终奖励
            rewards = {
                "progress_goal": raw_rewards["progress_goal"] * self.env.rew['progress_goal_reward_scale'],
                "gate_passed": raw_rewards["gate_passed"] * self.env.rew['gate_passed_reward_scale'],
                "crash": raw_rewards["crash"] * self.env.rew['crash_reward_scale'],
                "time": raw_rewards["time"] * self.env.rew['time_reward_scale'],
            }
            
            reward = torch.sum(torch.stack(list[Tensor](rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)
        else:
            # 测试时不计算奖励，返回零（我们只关心原始奖励值）
            reward = torch.zeros(self.num_envs, device=self.device)

        # 只在训练时进行日志记录（记录原始值，不受scale影响）
        if self.cfg.is_train:
            for key, value in raw_rewards.items():
                self._episode_sums[key] += value

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        # why so that we know where we are in the map and other frame to be applied to the drone?
        current_gate_idx = self._idx_wp  # Use strategy's state
        # make it one-hot for each environment
        one_hot_gate_idx = torch.zeros(self.num_envs, self.env._waypoints.shape[0], dtype=torch.float32, device=self.device)
        one_hot_gate_idx[torch.arange(self.num_envs), current_gate_idx] = 1.0

        # current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        current_gate_yaw = self.env._waypoints[current_gate_idx, -1].unsqueeze(-1)    # Yaw orientation of current gate (num_envs, 1)
        
        # Next gate information for planning ahead
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]  # World position of next gate
        next_gate_yaw = self.env._waypoints[next_gate_idx, -1].unsqueeze(-1)    # Yaw orientation of next gate (num_envs, 1)
        
        # Relative position to next gate in world frame
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_pos_next_gate_w = next_gate_pos_w - drone_pos_w  # Vector from drone to next gate

        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        # gate_pos_b, _ = subtract_frame_transforms(
        #     self.env._robot.data.root_link_pos_w,
        #     self.env._robot.data.root_quat_w,
        #     current_gate_pos_w
        # )

        # Previous actions
        prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        # gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                ### Basic drone states ###
                drone_ang_vel_b,    # angular velocity in the body frame (3 dims)
                drone_pose_w,       # position in the world frame (3 dims)
                drone_lin_vel_b,    # velocity in the body frame (3 dims)
                drone_quat_w,       # quaternion in the world frame (4 dims)
                ### for gate
                drone_pos_gate_frame,   # relative position to current gate in gate frame (3 dims)
                current_gate_yaw,      # yaw orientation of current gate (1 dim)
                # one_hot_gate_idx,    # one-hot encoded index of current gate (6 dims)
                ### for next gate planning ###
                # drone_pos_next_gate_w,  # relative position to next gate in world frame (3 dims)
                # next_gate_yaw,         # yaw orientation of next gate (1 dim)
                ### for dynamics adaptation ###
                prev_actions,       # previous actions (4 dims)
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                # extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s  # per second
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s * 60  # per minute
                extras["Episode_RewardSum/" + key] = episodic_sum_avg
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids) if env_ids is not None else self.num_envs
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        # start from the zeroth waypoint (beginning of the race)
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        # if self.cfg.is_train:
        if self.cfg.is_train:
            # Training mode: Random initialization in circular area around gate center
            # Calculate gate center (mean of all gates xy)
            gate_center_x = torch.mean(self.env._waypoints[:, 0])
            gate_center_y = torch.mean(self.env._waypoints[:, 1])
            
            # Calculate distance from center to farthest gate
            distances_to_center = torch.sqrt(
                (self.env._waypoints[:, 0] - gate_center_x) ** 2 + 
                (self.env._waypoints[:, 1] - gate_center_y) ** 2
            )
            max_distance = torch.max(distances_to_center)
            radius = max_distance * 1.25  # 1.25x multiplier for range
            
            # Generate random points in circle using polar coordinates
            r = torch.sqrt(torch.empty(n_reset, device=self.device).uniform_(0, 1)) * radius
            theta = torch.empty(n_reset, device=self.device).uniform_(0, 2 * torch.pi)
            
            # Convert to Cartesian coordinates
            initial_x = gate_center_x + r * torch.cos(theta)
            initial_y = gate_center_y + r * torch.sin(theta)
            # Random height between 0.3m and 2.5m (safe range based on death conditions: min_altitude=0.1m, max_altitude=3.0m)
            initial_z = torch.empty(n_reset, device=self.device).uniform_(0.04, 2.5)
            
            # Random initial waypoint (not necessarily the first gate)
            waypoint_indices = torch.randint(0, self.env._waypoints.shape[0], (n_reset,), device=self.device, dtype=self._idx_wp.dtype)
            
            # Point drone towards a random direction with some noise
            initial_yaw = torch.empty(n_reset, device=self.device).uniform_(0, 2 * torch.pi)
            
            # Set position
            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            
            # Set orientation
            quat = quat_from_euler_xyz(
                torch.zeros(n_reset, device=self.device),
                torch.zeros(n_reset, device=self.device),
                initial_yaw
            )
            default_root_state[:, 3:7] = quat
        else:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions (in strategy)
        self._idx_wp[env_ids] = waypoint_indices
        self._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()
        self._n_gates_passed[env_ids] = 0
        
        # Sync to env if it exists (for backward compatibility)
        if hasattr(self.env, '_idx_wp'):
            self.env._idx_wp[env_ids] = waypoint_indices
        if hasattr(self.env, '_desired_pos_w'):
            self.env._desired_pos_w[env_ids, :2] = self._desired_pos_w[env_ids, :2]
            self.env._desired_pos_w[env_ids, 2] = self._desired_pos_w[env_ids, 2]
        if hasattr(self.env, '_n_gates_passed'):
            self.env._n_gates_passed[env_ids] = 0
        if hasattr(self.env, '_last_distance_to_goal'):
            self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
                self._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
            )
        
        # Reset previous progress for reward calculation
        self._prev_progress_global[env_ids] = 0.0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        if hasattr(self.env, '_yaw_n_laps'):
            self.env._yaw_n_laps[env_ids] = 0

        if hasattr(self.env, '_pose_drone_wrt_gate'):
            self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
                self.env._waypoints[self._idx_wp[env_ids], :3],
                self.env._waypoints_quat[self._idx_wp[env_ids], :],
                self.env._robot.data.root_link_state_w[env_ids, :3]
            )

        if hasattr(self.env, '_prev_x_drone_wrt_gate'):
            self.env._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)

        if hasattr(self.env, '_crashed'):
            self.env._crashed[env_ids] = 0
        
        # Initialize previous position for gate passing detection (in strategy)
        self._prev_drone_pos_w[env_ids] = self.env._robot.data.root_link_pos_w[env_ids].clone()