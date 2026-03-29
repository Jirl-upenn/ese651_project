# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            None,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            observations,
            critic_observations,
            sampled_actions,
            value_targets,
            advantage_estimates,
            discounted_returns,
            prev_log_probs,
            prev_mean_actions,
            prev_action_stds,
            hidden_states,
            episode_masks,
            _,  # rnd_state_batch - not used anymore
        ) in generator:
            # TODO ----- START -----
            # 1. Evaluate current policy and value function using the saved observations
            if self.actor_critic.is_recurrent:
                # Handle Recurrent Neural Networks (RNNs/LSTMs) if enabled
                current_values = self.actor_critic.evaluate(critic_observations, hidden_states=hidden_states[1], masks=episode_masks).squeeze()
                current_log_probs, entropy = self.actor_critic.evaluate_actions(observations, sampled_actions, hidden_states=hidden_states[0], masks=episode_masks)
                current_log_probs = current_log_probs.squeeze()
            else:
                # Standard Feed-Forward Neural Network
                current_values = self.actor_critic.evaluate(critic_observations).squeeze()

                # 1. First, tell the actor to build the distribution based on current observations
                self.actor_critic.update_distribution(observations)
                # 2. Get the log probability of the actions taken
                current_log_probs = self.actor_critic.get_actions_log_prob(sampled_actions).squeeze()
                # 3. Get the entropy (using the @property you have in actor_critic.py)
                entropy = self.actor_critic.entropy

            # --- PREVENT TENSOR BROADCASTING BUG ---
            # This forces all tensors to be flat 1D lists so they multiply line-by-line
            prev_log_probs = prev_log_probs.squeeze()
            advantage_estimates = advantage_estimates.squeeze()
            # Advantage Normalization
            advantage_estimates = (advantage_estimates - advantage_estimates.mean()) / (advantage_estimates.std() + 1e-8)

            value_targets = value_targets.squeeze()
            discounted_returns = discounted_returns.squeeze()
            # ---------------------------------------

            # 2. Calculate Surrogate (Actor) Loss
            # Ratio: exp(current_log_prob - prev_log_prob)
            ratio = torch.exp(current_log_probs - prev_log_probs)
            
            # Unclipped and Clipped surrogate objectives
            surrogate_loss_1 = advantage_estimates * ratio
            surrogate_loss_2 = advantage_estimates * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            
            # We want to MAXIMIZE the surrogate objective, so we MINIMIZE its negative
            surrogate_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()

            # 3. Calculate Value (Critic) Loss
            if self.use_clipped_value_loss:
                # Clip the value estimate to prevent too large of a value update
                value_error_unclipped = (current_values - discounted_returns) ** 2
                current_values_clipped = value_targets + torch.clamp(
                    current_values - value_targets, -self.clip_param, self.clip_param
                )
                value_error_clipped = (current_values_clipped - discounted_returns) ** 2
                value_loss = 0.5 * torch.max(value_error_unclipped, value_error_clipped).mean()
            else:
                value_loss = 0.5 * ((current_values - discounted_returns) ** 2).mean()

            # 4. Total Loss Computation
            # Final loss = Actor Loss + (Value_Coef * Critic Loss) - (Entropy_Coef * Entropy)
            entropy_loss = entropy.mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

            # 5. Backpropagation step
            self.optimizer.zero_grad() # Clear old gradients
            loss.backward()            # Compute new gradients
            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()      # Update the neural network weights

            # ------------------------------------------------------------------
            # ADD (Adaptive Learning Rate Scheduler)
            # ------------------------------------------------------------------
            if self.schedule in ["adaptive", "empirical"]:
                with torch.no_grad():
                    # Calculate Approximate KL Divergence
                    # Formula: log_prob_old - log_prob_new
                    kl_divergence = (prev_log_probs - current_log_probs).mean().item()

                # If the policy has changed too drastically (taken too large a step, at risk of falling over) -> reduce learning rate
                if kl_divergence > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                # If the policy has almost not changed (learning too slowly) -> increase learning rate
                elif kl_divergence < self.desired_kl / 2.0 and kl_divergence >= 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                # Apply the calculated new learning rate to the PyTorch optimizer
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
            # ------------------------------------------------------------------

            # 6. Accumulate metrics for tracking (Wandb/Terminal)
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_loss.item()
            # TODO ----- END -----

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # Clear the storage
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy
