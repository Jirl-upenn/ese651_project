# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

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
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
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

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # 1. GATE DETECTION: Plane Crossing Detection
        curr_gate_idx = self.env._idx_wp
        drone_pos = self.env._robot.data.root_link_pos_w

        gate_pos = self.env._waypoints[curr_gate_idx, :3]
        gate_quat = self.env._waypoints_quat[curr_gate_idx, :]
        # Calculate the gate's position relative to the drone's body frame (B) for accurate crossing detection
        local_pos_b, _ = subtract_frame_transforms(
            gate_pos, 
            gate_quat, 
            drone_pos, 
            self.env._robot.data.root_quat_w
        )

        curr_local_x = local_pos_b[:, 0]
        curr_local_y = local_pos_b[:, 1]
        curr_local_z = local_pos_b[:, 2]

        # Logic for detecting if the drone has just crossed the plane of the gate:
        # crossed_forward = (self.env._prev_x_drone_wrt_gate < 0.0) & (curr_local_x >= 0.0)
        crossed_forward = (self.env._prev_x_drone_wrt_gate > 0.0) & (curr_local_x <= 0.0)
        # To prevent false positives from drones that are very far away and just happen to cross the plane, we add a distance check.
        valid_distance = torch.abs(self.env._prev_x_drone_wrt_gate) < 1.5
        crossed_plane = crossed_forward & valid_distance

        alpha = self.env._prev_x_drone_wrt_gate / (self.env._prev_x_drone_wrt_gate - curr_local_x + 1e-8)
        alpha = torch.clamp(alpha, 0.0, 1.0)

        cross_y = self.env._prev_y_drone_wrt_gate + alpha * (curr_local_y - self.env._prev_y_drone_wrt_gate)
        cross_z = self.env._prev_z_drone_wrt_gate + alpha * (curr_local_z - self.env._prev_z_drone_wrt_gate)

        # Logic for checking if the drone is within the gate boundaries
        in_gate_y = torch.abs(cross_y) < 0.45
        in_gate_z = torch.abs(cross_z) < 0.45

        # Final gate passage condition: must cross the plane and be within the gate boundaries
        gate_passed = crossed_plane & in_gate_y & in_gate_z

        # Before we update the prev_x for the next step, we need to handle the case where a drone has just passed through the gate.
        self.env._prev_x_drone_wrt_gate = curr_local_x.clone()
        self.env._prev_y_drone_wrt_gate = curr_local_y.clone()
        self.env._prev_z_drone_wrt_gate = curr_local_z.clone()

        # Handle gate passage: For drones that passed through the gate, we need to:
        ids_gate_passed = torch.where(gate_passed)[0]
        if len(ids_gate_passed) > 0:
            # 1. Increment the waypoint index to point to the next gate in the track
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

            # 2. Update the desired position for those drones to the next gate's position 
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

            # 3. Reset the prev_x_drone_wrt_gate for those drones to prevent false gate passages in the next steps. We can set it to the current local x position relative to the new gate, which we will calculate now:
            new_gate_pos = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
            new_gate_quat = self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :]
            
            new_local_pos_b, _ = subtract_frame_transforms(
                new_gate_pos, 
                new_gate_quat, 
                drone_pos[ids_gate_passed], 
                self.env._robot.data.root_quat_w[ids_gate_passed] # 补齐参数
            )
            # Set the previous x position relative to the new gate
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_local_pos_b[:, 0].clone()
            self.env._prev_y_drone_wrt_gate[ids_gate_passed] = new_local_pos_b[:, 1].clone()
            self.env._prev_z_drone_wrt_gate[ids_gate_passed] = new_local_pos_b[:, 2].clone()

        # 2. PRO RACING PROGRESS
        # Reward for making progress towards the gate, measured as the speed along the direction to the gate (encourages fast and direct flying towards the gate)
        drone_to_gate_w = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        dist_to_gate = torch.linalg.norm(drone_to_gate_w, dim=1, keepdim=True) + 1e-8
        dir_to_gate = drone_to_gate_w / dist_to_gate
        # speed along the direction to the gate (dot product of velocity and direction to gate)
        drone_vel = self.env._robot.data.root_lin_vel_w
        progress_speed = torch.sum(drone_vel * dir_to_gate, dim=1)
        # Clamp the progress reward to prevent large spikes, and scale it down
        progress = torch.clamp(progress_speed, min=-10.0, max=10.0) * 0.2

        # Add a small penalty for changing actions too abruptly, to encourage smoother flying (but don't penalize it too much or it won't learn power loops!)
        action_diff = torch.sum(torch.square(self.env._actions - self.env._previous_actions), dim=1)
        # Spin Penalty
        ang_vel = self.env._robot.data.root_ang_vel_w
        spin_penalty = torch.sum(torch.square(ang_vel), dim=1) * 0.01
        # Time penalty
        time_penalty = 0.005 # 0.005
        # Bonus for passing through the gate
        progress = progress + (gate_passed.float() * 10.0) - (action_diff * 0.005) - spin_penalty - time_penalty

        # -------------------------------------------------------------
        # DELETED: self.env._last_distance_to_goal (No longer needed!)
        # DELETED: tilt_penalty (Must be removed to allow power loops!)
        # -------------------------------------------------------------

        # 3. CRASH DETECTION (Give it a tiny grace period to spawn)
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 10).int() 
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim."""

        # 1. Drone's own states (Ego-centric)
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b  # Forward/side/up speeds
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b      # Roll/pitch/yaw spin rates
        
        # 🌟 新增：机身坐标系下的重力向量（告诉它哪边是地，做 Power Loop 必备！）
        # IsaacLab 的 robot.data 通常直接提供 projected_gravity_b
        gravity_b = self.env._robot.data.projected_gravity_b 

        # 2. Where is the CURRENT gate and how is it oriented?
        current_gate_idx = self.env._idx_wp
        current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]
        current_gate_quat_w = self.env._waypoints_quat[current_gate_idx, :] # 获取门的朝向
        
        # 计算当前门相对于无人机机身的【位置】和【朝向】
        gate_pos_b, gate_quat_b = subtract_frame_transforms(
            current_gate_pos_w,
            current_gate_quat_w,
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w
        )
        
        # 🌟 新增：Where is the NEXT gate? (Lookahead，帮助规划赛车线)
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        
        next_gate_pos_b, _ = subtract_frame_transforms(
            next_gate_pos_w,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1), # 假设暂不需要下一个门的朝向，随便传个默认四元数
            self.env._robot.data.root_link_pos_w,
            self.env._robot.data.root_quat_w
        )

        # 3. What was I just doing?
        prev_actions = self.env._previous_actions

        obs = torch.cat(
            [
                drone_lin_vel_b,    # (3) How fast am I moving?
                drone_ang_vel_b,    # (3) How fast am I spinning?
                gravity_b,          # (3) 🌟 Which way is down? (Crucial for attitude)
                gate_pos_b,         # (3) Where is the center of the current gate?
                gate_quat_b,        # (4) 🌟 Which way is the current gate facing?
                next_gate_pos_b,    # (3) 🌟 Where is the next gate? (Lookahead)
                prev_actions        # (4) What were my last motor commands?
            ],
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
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
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

        n_reset = len(env_ids)
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
        # 1. Pick a RANDOM gate to start at so it learns the whole track simultaneously
        waypoint_indices = torch.randint(0, self.env._waypoints.shape[0], (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype)

        # Get base positions of those random gates
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        # 2. Add Domain Randomization (Spawn them slightly offset and messy)
        x_local = torch.empty(n_reset, device=self.device).uniform_(-4.0, -0.5) # Spawn 0.5m to 6.0m behind gate (Widen the spawn distance so it learns to fly from far away!)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)  # Shifted left/right
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)  # Shifted up/down

        # Rotate local pos to global frame
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

        # 3. Point drone towards the gate, but add random tilt/yaw noise so it learns to stabilize
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        quat = quat_from_euler_xyz(
            torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2), # Random Roll
            torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2), # Random Pitch
            initial_yaw + torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3) # Random Yaw
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
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

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        # self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
        #     self.env._waypoints[self.env._idx_wp[env_ids], :3],
        #     self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
        #     self.env._robot.data.root_link_state_w[env_ids, :3]
        # )
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            default_root_state[:, :3],
            default_root_state[:, 3:7]  
        )

        # self.env._prev_x_drone_wrt_gate[env_ids] = 1.0
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids][:, 0].clone()
        self.env._prev_y_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids][:, 1].clone()
        self.env._prev_z_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids][:, 2].clone()
        self.env._crashed[env_ids] = 0

'''
conda activate env_isaaclab
export PYTHONPATH=$(pwd)



python scripts/rsl_rl/train_race.py \
--task Isaac-Quadcopter-Race-v0 \
--num_envs 4096 \
--max_iterations 1000 \
--headless \
--logger wandb     

'''