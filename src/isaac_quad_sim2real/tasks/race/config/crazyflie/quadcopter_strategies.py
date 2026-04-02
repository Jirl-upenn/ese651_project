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
        # Added local variables 
        self._prev_y_drone_wrt_gate = torch.zeros(self.num_envs, device=self.device)
        self._prev_z_drone_wrt_gate = torch.zeros(self.num_envs, device=self.device)
        self._gate_passed_wrong_way = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._segment_start_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

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

        # ==========================================================
        # 🚨 DOMAIN RANDOMIZATION BOUNDS
        # ==========================================================
        # # 1. TWR
        # self._twr_min = self.env._twr_value * 0.9
        # self._twr_max = self.env._twr_value * 1.1
        
        # # 2. Aerodynamics
        # self._k_aero_xy_min = self.env._k_aero_xy_value * 0.45
        # self._k_aero_xy_max = self.env._k_aero_xy_value * 2.2
        # self._k_aero_z_min = self.env._k_aero_z_value * 0.45
        # self._k_aero_z_max = self.env._k_aero_z_value * 2.2
        
        # # 3. PID gains (Roll/Pitch)
        # self._kp_omega_rp_min = self.env._kp_omega_rp_value * 0.8
        # self._kp_omega_rp_max = self.env._kp_omega_rp_value * 1.25
        # self._ki_omega_rp_min = self.env._ki_omega_rp_value * 0.8
        # self._ki_omega_rp_max = self.env._ki_omega_rp_value * 1.25
        # self._kd_omega_rp_min = self.env._kd_omega_rp_value * 0.65
        # self._kd_omega_rp_max = self.env._kd_omega_rp_value * 1.4
        
        # # 4. PID gains (Yaw)
        # self._kp_omega_y_min = self.env._kp_omega_y_value * 0.8
        # self._kp_omega_y_max = self.env._kp_omega_y_value * 1.25
        # self._ki_omega_y_min = self.env._ki_omega_y_value * 0.8
        # self._ki_omega_y_max = self.env._ki_omega_y_value * 1.25
        # self._kd_omega_y_min = self.env._kd_omega_y_value * 0.65
        # self._kd_omega_y_max = self.env._kd_omega_y_value * 1.4
        # ==========================================================
        # 1. TWR
        self._twr_min = self.env._twr_value * 0.95
        self._twr_max = self.env._twr_value * 1.05
        
        # 2. Aerodynamics
        self._k_aero_xy_min = self.env._k_aero_xy_value * 0.5
        self._k_aero_xy_max = self.env._k_aero_xy_value * 2.0
        self._k_aero_z_min = self.env._k_aero_z_value * 0.5
        self._k_aero_z_max = self.env._k_aero_z_value * 2.0
        
        # 3. PID gains (Roll/Pitch)
        self._kp_omega_rp_min = self.env._kp_omega_rp_value * 0.85
        self._kp_omega_rp_max = self.env._kp_omega_rp_value * 1.15
        self._ki_omega_rp_min = self.env._ki_omega_rp_value * 0.85
        self._ki_omega_rp_max = self.env._ki_omega_rp_value * 1.15
        self._kd_omega_rp_min = self.env._kd_omega_rp_value * 0.7
        self._kd_omega_rp_max = self.env._kd_omega_rp_value * 1.3
        
        # 4. PID gains (Yaw)
        self._kp_omega_y_min = self.env._kp_omega_y_value * 0.85
        self._kp_omega_y_max = self.env._kp_omega_y_value * 1.15
        self._ki_omega_y_min = self.env._ki_omega_y_value * 0.85
        self._ki_omega_y_max = self.env._ki_omega_y_value * 1.15
        self._kd_omega_y_min = self.env._kd_omega_y_value * 0.4
        self._kd_omega_y_max = self.env._kd_omega_y_value * 1.3
        # ==========================================================
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
        # Calculate drone position in the gate's local frame (with gate as origin and orientation)
        local_pos_b, _ = subtract_frame_transforms(
            gate_pos, 
            gate_quat, 
            drone_pos, 
            self.env._robot.data.root_quat_w
        )
        #
        curr_local_x = local_pos_b[:, 0]
        curr_local_y = local_pos_b[:, 1]
        curr_local_z = local_pos_b[:, 2]
        #
        prev_x = self.env._prev_x_drone_wrt_gate
        prev_y = self._prev_y_drone_wrt_gate
        prev_z = self._prev_z_drone_wrt_gate
        
        # Logic for detecting if the drone has just crossed the plane of the gate:
        crossed_forward = (prev_x > 0.0) & (curr_local_x <= 0.0)
        crossed_backward = (prev_x < 0.0) & (curr_local_x >= 0.0)

        # To prevent false positives from drones that are very far away and just happen to cross the plane, we add a distance check.
        valid_distance = torch.abs(self.env._prev_x_drone_wrt_gate) < 2.5

        #
        alpha = prev_x / (prev_x - curr_local_x + 1e-8)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        #
        cross_y = prev_y + alpha * (curr_local_y - prev_y)
        cross_z = prev_z + alpha * (curr_local_z - prev_z)

        # # Determine if the drone is currently heading towards gate 3 (for special reward shaping on that gate)
        # heading_to_gate_1 = (self.env._idx_wp == 1)
        # heading_to_gate_2 = (self.env._idx_wp == 2)
        
        # #  
        # gate_threshold = torch.where(heading_to_gate_1 | heading_to_gate_2, 0.45, 0.55)
        # # Apply the dynamic threshold instead of hardcoded 0.5
        in_gate = (
            (torch.abs(cross_y) < 0.7) & 
            (torch.abs(cross_z) < 0.7)
        )
        #
        in_gate_wrong_way= (
            (torch.abs(cross_y) < 0.7) & 
            (torch.abs(cross_z) < 0.7)
        )

        # Final gate passage condition: must cross the plane and be within the gate boundaries
        gate_passed = crossed_forward & valid_distance & in_gate
        # Detect if it passed through the gate in the wrong direction (for potential penalty)
        gate_passed_wrong_way = crossed_backward & valid_distance & in_gate_wrong_way
        # Store the gate passage info in the environment for use in rewards and logging
        self._gate_passed_wrong_way = gate_passed_wrong_way

        # Update previous local positions for the next step's crossing detection
        self.env._prev_x_drone_wrt_gate = curr_local_x.clone()
        self._prev_y_drone_wrt_gate = curr_local_y.clone()
        self._prev_z_drone_wrt_gate = curr_local_z.clone()
        # Handle gate passage: For drones that passed through the gate, we need to:
        ids_gate_passed = torch.where(gate_passed)[0]
        if len(ids_gate_passed) > 0:
            # 1. Increment the waypoint index to point to the next gate in the track
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]

            # 2. Update the desired position for those drones to the next gate's position 
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

            self._segment_start_pos[ids_gate_passed] = gate_pos[ids_gate_passed].clone()
            
            # 3. Reset the prev_x_drone_wrt_gate for those drones to prevent false gate passages in the next steps. We can set it to the current local x position relative to the new gate, which we will calculate now:
            new_gate_pos = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]
            new_gate_quat = self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :]
            
            new_local_pos_b, _ = subtract_frame_transforms(
                new_gate_pos, 
                new_gate_quat, 
                drone_pos[ids_gate_passed], 
                self.env._robot.data.root_quat_w[ids_gate_passed] 
            )
            # Set the previous x position relative to the new gate
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_local_pos_b[:, 0].clone()
            self._prev_y_drone_wrt_gate[ids_gate_passed] = new_local_pos_b[:, 1].clone()
            self._prev_z_drone_wrt_gate[ids_gate_passed] = new_local_pos_b[:, 2].clone()
            # Update current local positions
            curr_local_x[ids_gate_passed] = new_local_pos_b[:, 0].clone()
            curr_local_y[ids_gate_passed] = new_local_pos_b[:, 1].clone()
            curr_local_z[ids_gate_passed] = new_local_pos_b[:, 2].clone()
            

        # 2. PRO RACING PROGRESS
        # ==================================================================
        # NEW: Blended Progress (Optimal Racing Line + Gate Attraction)
        # ==================================================================
        drone_pos = self.env._robot.data.root_link_pos_w
        drone_vel = self.env._robot.data.root_lin_vel_w
        
        # 1. Prev Gate to current gate Direction
        track_segment_w = self.env._desired_pos_w - self._segment_start_pos
        dist_segment = torch.linalg.norm(track_segment_w, dim=1, keepdim=True) + 1e-8
        segment_dir = track_segment_w / dist_segment
        # 2. Drone to the gate direction
        drone_to_gate_w = self.env._desired_pos_w - drone_pos
        dist_to_gate = torch.linalg.norm(drone_to_gate_w, dim=1, keepdim=True) + 1e-8
        dir_to_gate = drone_to_gate_w / dist_to_gate
        # 3. Blended direction: 70% along the optimal racing line (segment_dir) and 30% towards the gate (dir_to_gate)
        # blended_dir = (0.1 * segment_dir) + (0.9 * dir_to_gate)
        blended_dir= dir_to_gate
        blended_dir = blended_dir / torch.linalg.norm(blended_dir, dim=1, keepdim=True) 

        # 4. 最终速度投影
        progress_speed = torch.sum(drone_vel * blended_dir, dim=1)
        # ==================================================================

        # ------------------------------------------------------------------
        # Waypoint 3 Reward Shaping
        # ------------------------------------------------------------------
        heading_to_gate_3 = (self.env._idx_wp == 3)
        # 1. Calculate horizontal progress (X and Y axes)
        drone_to_gate_xy = drone_to_gate_w[:, :2]
        dist_to_gate_xy = torch.linalg.norm(drone_to_gate_xy, dim=1, keepdim=True) + 1e-8
        dir_to_gate_xy = drone_to_gate_xy / dist_to_gate_xy
        progress_xy = torch.sum(drone_vel[:, :2] * dir_to_gate_xy, dim=1)

        # 2. Calculated custom progress for gate 3
        # gate3_custom_progress = (progress_xy * 1.5) + (progress_z * 1.0)
        progress_speed = torch.where(heading_to_gate_3, progress_xy, progress_speed)

        # 3. Relax the penalty for flying backward/inverted over the top
        # is_negative_progress = progress_speed < 0
        # mask_relax_penalty = heading_to_gate_3 & is_negative_progress
        # # progress_speed[mask_relax_penalty] = progress_speed[mask_relax_penalty] * 0.5
        # progress_speed[mask_relax_penalty] = progress_speed[mask_relax_penalty] * 0.1

        # 4. Add a bonus for climbing up during the approach to gate 3 to encourage power loops 
        # Calculate how high the drone is relative to the gate
        relative_height = drone_pos[:, 2] - self.env._desired_pos_w[:, 2]
        # 
        is_in_front_of_gate = (curr_local_x < 0.0)
        # Only True if the drone is heading to Gate 3 AND is less than 2.0 meters above it
        is_climbing_phase = heading_to_gate_3 & (relative_height < 1) & is_in_front_of_gate# cant excute power loop w 0.8
        # Reward positive Z velocity, but clamp negatives to 0 so we don't punish diving
        z_vel_reward = torch.clamp(drone_vel[:, 2], min=0.0, max=3.0)

        # Apply the reward ONLY during the climbing phase
        climb_bonus = torch.where(is_climbing_phase, z_vel_reward * 0.8, 0.0)
        # climb_bonus = torch.where(is_climbing_phase, z_vel_reward * 0.8, 0.0)
        progress_speed += climb_bonus
        # ------------------------------------------------------------------

        # Clamp the progress reward to prevent large spikes, and scale it down
        progress = torch.clamp(progress_speed, min=-10.0, max=20.0) * 0.2

        abs_speed = torch.linalg.norm(drone_vel, dim=1)
        speed_bonus = abs_speed * 0.05

        # Add a small penalty for changing actions too abruptly, to encourage smoother flying (but don't penalize it too much or it won't learn power loops!)
        # action_diff = torch.sum(torch.square(self.env._actions - self.env._previous_actions), dim=1) * 0.005
        # Spin Penalty
        ang_vel = self.env._robot.data.root_ang_vel_b

        # 1. A simple spin penalty that penalizes all angular velocities equally (not ideal for racing, as it would discourage power loops)
        # spin_penalty = torch.sum(torch.square(ang_vel), dim=1) * 0.01

        # 2. A more nuanced spin penalty that penalizes pitch (Y-axis) less than roll and yaw, to allow for power loops (but still penalizes excessive spinning in all axes)
        # ang_vel_weights = torch.tensor([1.0, 0.6, 1.0], device=self.device) # Penalize pitch less for power loops
        # spin_penalty = torch.sum(ang_vel_weights * torch.square(ang_vel), dim=1) * 0.01

        # 3. An adaptive spin penalty that penalizes pitch (Y-axis) less only when the drone is flying towards Gate 3, to allow for power loops on that specific gate (but still penalizes excessive spinning in all axes and at other gates)
        ang_vel_weights = torch.ones((self.num_envs, 3), device=self.device)
        # heading_to_gate_3 = (self.env._idx_wp == 2)
        ang_vel_weights[heading_to_gate_3] = torch.tensor([1.0, 0.05, 1.0], device=self.device) 
        spin_penalty = torch.sum(ang_vel_weights * torch.square(ang_vel), dim=1) * 0.002 #0.01 -> 0.005 -> 0.002

        spin_penalty = torch.clamp(spin_penalty, max=2.0) #
        # Time penalty
        time_penalty = torch.ones_like(progress) * 0.25 # 0.005 -> 0.05 -> 0.15 -> 0.35 -> 0.23 -> 0.1
        # Bonus for passing through the gate


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
                "speed_bonus": speed_bonus * self.env.rew['progress_goal_reward_scale'],
                "gate_passed": (gate_passed.float() * 10.0) * self.env.rew['progress_goal_reward_scale'],
                # "penalty_action": -1 * action_diff * self.env.rew['progress_goal_reward_scale'],
                "penalty_spin": -1 * spin_penalty * self.env.rew['progress_goal_reward_scale'],
                "penalty_time": -1 * time_penalty * self.env.rew['progress_goal_reward_scale'],
                "crash": crashed * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            # for key, value in rewards.items():
            #     self._episode_sums[key] += value
            for key, value in rewards.items():
                # 如果这个键在计分板里不存在，就自动为所有的无人机创建一个全 0 的记录器！
                if hasattr(self, '_episode_sums'):
                    if key not in self._episode_sums:
                        self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
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
        # gate_pos_b, gate_quat_b = subtract_frame_transforms(
        #     current_gate_pos_w,
        #     current_gate_quat_w,
        #     self.env._robot.data.root_link_pos_w,
        #     self.env._robot.data.root_quat_w
        # )
        
        gate_pos_b, gate_quat_b = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,  # 🌟 A: Drone
            self.env._robot.data.root_quat_w,
            current_gate_pos_w,                    # 🌟 B: Gate
            current_gate_quat_w
        )

        # 🌟 新增：Where is the NEXT gate? (Lookahead，帮助规划赛车线)
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_gate_idx, :3]
        # next_gate_quat_w = self.env._waypoints_quat[next_gate_idx, :] # 获取下一个门的朝向
        
        # next_gate_pos_b, next_gate_quat_b = subtract_frame_transforms(
        #     self.env._robot.data.root_link_pos_w,  # 🌟 A: Drone
        #     self.env._robot.data.root_quat_w,
        #     next_gate_pos_w,                       # 🌟 B: Next Gate
        #     next_gate_quat_w
        # )
        next_gate_pos_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_pos_w,  # 🌟 A: Drone
            self.env._robot.data.root_quat_w,
            next_gate_pos_w,                       # 🌟 B: Next Gate
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
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
                # next_gate_quat_b,   # (4) 🌟 Which way is the next gate facing?
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
        # z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)  # Shifted up/down
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.7, 0.7)  # Shifted up/down
        # Rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        
        # ==========================================================
        # Mix it up with some ground starts 
        # ==========================================================
        # Find out which resets are spawning at the initial gate 
        is_initial_gate = (waypoint_indices == self.env._initial_wp)
        # Set 30% of those initial gate spawns to be on the ground 
        is_ground_start = is_initial_gate & (torch.rand(n_reset, device=self.device) < 0.3)
        z_air = z_local + z_wp
        z_ground = 0.05  # 
        # If it's a ground start, set z to 0.05m, otherwise use the randomized air spawn height
        initial_z = torch.where(is_ground_start, z_ground, z_air)
        # For ground starts, set the initial velocities and orientations to 0.
        default_root_state[is_ground_start, 7:13] = 0.0
        # ==========================================================

        # # ==========================================================
        # # 🌪️ Power Loop 特训：强制 wp3 的生成点回到 wp2 的前方
        # # ==========================================================
        # loop_target_idx = 3  # wp3 (目标门)
        # loop_start_idx = 2   # wp2 (Power Loop 的起点门)
        
        # is_loop_target = (waypoint_indices == loop_target_idx)
        # n_loop = int(torch.count_nonzero(is_loop_target).item())

        # if n_loop > 0:
        #     # 获取 wp2 的坐标和朝向
        #     x0_wp2 = self.env._waypoints[loop_start_idx, 0]
        #     y0_wp2 = self.env._waypoints[loop_start_idx, 1]
        #     z_wp2  = self.env._waypoints[loop_start_idx, 2]
        #     theta_wp2 = self.env._waypoints[loop_start_idx, -1]

        #     # 1. 局部坐标：因为你的代码里负数是门后，所以这里用正数（0.3 到 2.0）让它生成在 wp2 的门前（已经穿过 wp2）
        #     x_local_pl = torch.empty(n_loop, device=self.device).uniform_(0.3, 2.0) 
        #     y_local_pl = torch.empty(n_loop, device=self.device).uniform_(-1.0, 0.7)
        #     z_local_pl = torch.empty(n_loop, device=self.device).uniform_(-0.7, 0.7)

        #     # 2. 
        #     cos_theta_wp2 = torch.cos(theta_wp2)
        #     sin_theta_wp2 = torch.sin(theta_wp2)
        #     x_rot_pl = cos_theta_wp2 * x_local_pl - sin_theta_wp2 * y_local_pl
        #     y_rot_pl = sin_theta_wp2 * x_local_pl + cos_theta_wp2 * y_local_pl
        #     initial_x[is_loop_target] = x0_wp2 - x_rot_pl
        #     initial_y[is_loop_target] = y0_wp2 - y_rot_pl
        #     initial_z[is_loop_target] = z_wp2 + z_local_pl
            
        #     # 4. 修复机头朝向：让它顺着 wp2 的方向准备进入 Power Loop
        #     # 我们需要覆盖这部分无人机的 initial_yaw
        #     initial_yaw_pl = torch.atan2(y0_wp2 - initial_y[is_loop_target], x0_wp2 - initial_x[is_loop_target])
        #     # 但因为它是从 wp2 往前飞，它的机头应该是背离 wp2 的，所以可能需要加上 pi (根据你的坐标系定)
        #     # 或者更简单直接的方法：直接让机头朝向顺着 wp2 的 theta 加上一点随机扰动
        #     yaw_pl = theta_wp2 + torch.empty(n_loop, device=self.device).uniform_(-0.3, 0.3)
            
        #     quat_pl = quat_from_euler_xyz(
        #         torch.empty(n_loop, device=self.device).uniform_(-0.2, 0.2), # Roll
        #         torch.empty(n_loop, device=self.device).uniform_(-0.2, 0.2), # Pitch
        #         yaw_pl # 修正后的 Yaw
        #     )
        #     # 覆盖 quat
        #     quat[is_loop_target] = quat_pl
        # # ==========================================================

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

        self._segment_start_pos[env_ids, 0] = initial_x.clone()
        self._segment_start_pos[env_ids, 1] = initial_y.clone()
        self._segment_start_pos[env_ids, 2] = initial_z.clone()

        # ==========================================================
        # 🚨 DYNAMICS DOMAIN RANDOMIZATION
        # ==========================================================
        if self.cfg.is_train: 
            # 1. Randomize TWR
            self.env._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(self._twr_min, self._twr_max)
            # 2. Randomize Aerodynamics
            self.env._K_aero[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(self._k_aero_xy_min, self._k_aero_xy_max)
            self.env._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(self._k_aero_z_min, self._k_aero_z_max)
            # 3. Randomize PID Gains (Roll/Pitch)
            self.env._kp_omega[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(self._kp_omega_rp_min, self._kp_omega_rp_max)
            self.env._ki_omega[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(self._ki_omega_rp_min, self._ki_omega_rp_max)
            self.env._kd_omega[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(self._kd_omega_rp_min, self._kd_omega_rp_max)
            # 4. Randomize PID Gains (Yaw)
            self.env._kp_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(self._kp_omega_y_min, self._kp_omega_y_max)
            self.env._ki_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(self._ki_omega_y_min, self._ki_omega_y_max)
            self.env._kd_omega[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(self._kd_omega_y_min, self._kd_omega_y_max)
        # ==========================================================

        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            # x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            # y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)
            
            # x_local and y_local are FIXED to the center
            x_local = torch.tensor([-1.75], device=self.device)
            y_local = torch.tensor([0.0], device=self.device)

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
        self._prev_y_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids][:, 1].clone()
        self._prev_z_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids][:, 2].clone()

        self.env._crashed[env_ids] = 0
        self._gate_passed_wrong_way[env_ids] = False
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