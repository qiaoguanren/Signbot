# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# MDPO version: adapted from your PPO-Penalty code.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage, ReplayBuffer
import wandb
from rsl_rl.utils import unpad_trajectories
import time


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + (delta**2) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class MDPOMimic:
    def __init__(
        self,
        env,
        actor_critic,
        estimator,
        estimator_paras,
        depth_encoder,
        depth_encoder_paras,
        depth_actor,
        student_actor,
        distill_paras,
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
        device='cpu',
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0, 0, 0],  
        **kwargs
    ):

        self.env = env
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate


        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = 1
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss

        self.beta = 1.0
        self.adaptive_beta = True
        self.total_iterations = 2.5e4
        self.iter_count = 0

        self.hist_encoder_optimizer = optim.Adam(
            self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate
        )
        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0

        # Estimator
        self.estimator = estimator
        self.priv_states_dim = estimator_paras["priv_states_dim"]
        self.est_start = estimator_paras["priv_start"]
        self.num_prop = estimator_paras["prop_dim"]
        self.prop_start = estimator_paras["prop_start"]

        self.estimator_optimizer = optim.Adam(
            self.estimator.parameters(), lr=estimator_paras["learning_rate"]
        )
        self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # Depth encoder
        self.if_depth = depth_encoder is not None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_optimizer = optim.Adam(
                self.depth_encoder.parameters(),
                lr=depth_encoder_paras["learning_rate"],
            )
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam(
                [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                lr=depth_encoder_paras["learning_rate"],
            )

        # Student actor
        self.if_distill = student_actor is not None
        if self.if_distill:
            self.student_actor = student_actor
            self.student_actor_optimizer = optim.Adam(
                self.student_actor.parameters(), lr=distill_paras["learning_rate"]
            )

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        if self.train_with_estimated_states:
            obs_est = obs.clone()
            priv_states_estimated = self.estimator(
                obs_est[:, self.prop_start : self.prop_start + self.num_prop]
            )
            obs_est[
                :,
                self.est_start : self.est_start + self.priv_states_dim
            ] = priv_states_estimated
            self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()
        else:
            self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += (
                self.gamma
                * torch.squeeze(
                    self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device),
                    1,
                )
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
        return rewards

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimator_loss = 0.0
        mean_discriminator_loss = 0.0
        mean_discriminator_acc = 0.0
        mean_priv_reg_loss = 0.0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for sample in generator:
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            ) = sample

            obs_est_batch = obs_batch.clone()
            priv_states_predicted = self.estimator(
                obs_batch[:, self.prop_start : self.prop_start + self.num_prop]
            )
            obs_est_batch[
                :, self.est_start : self.est_start + self.priv_states_dim
            ] = priv_states_predicted.detach()

            self.actor_critic.act(
                obs_est_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch,
                masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(
                p=2, dim=1
            ).mean()


            priv_reg_stage = (
                min(
                    max((self.counter - self.priv_reg_coef_schedual[2]), 0)
                    / self.priv_reg_coef_schedual[3],
                    1,
                )
                if len(self.priv_reg_coef_schedual) > 3
                else 0.0
            )
            priv_reg_coef = (
                priv_reg_stage
                * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0])
                + self.priv_reg_coef_schedual[0]
            )

            estimator_loss = (
                priv_states_predicted
                - obs_batch[:, self.est_start : self.est_start + self.priv_states_dim]
            ).pow(2).mean()

            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(
                self.estimator.parameters(), self.max_grad_norm
            )
            self.estimator_optimizer.step()


            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate_loss = -(torch.squeeze(advantages_batch) * ratio).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (value_batch - returns_batch).pow(2).mean()

            kl = torch.sum(
                torch.log(sigma_batch / (old_sigma_batch + 1e-8) + 1e-8)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                  / (2.0 * torch.square(sigma_batch) + 1e-8)
                - 0.5,
                axis=-1
            )
            kl_mean = kl.mean()

            if self.schedule == 'adaptive' and self.desired_kl is not None and self.adaptive_beta:
                if kl_mean > self.desired_kl * 1.5:
                    self.beta *= 2.0
                elif kl_mean < self.desired_kl / 1.5:
                    self.beta /= 2.0

            loss = (
                surrogate_loss \
                + self.value_loss_coef * value_loss \
                # - self.entropy_coef * entropy_batch.mean() \
                + priv_reg_coef * priv_reg_loss \
                + self.beta * kl_mean
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates

        mean_discriminator_loss = 0.0
        mean_discriminator_acc = 0.0

        self.storage.clear()
        self.update_counter()

        self.iter_count += 1

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_estimator_loss,
            mean_discriminator_loss,
            mean_discriminator_acc,
            mean_priv_reg_loss,
            priv_reg_coef,
        )

    def update_dagger(self):
        mean_hist_latent_loss = 0.0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            with torch.inference_mode():
                self.actor_critic.act(
                    obs_batch,
                    hist_encoding=True,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )

            with torch.inference_mode():
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
            hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)

            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(
                p=2, dim=1
            ).mean()

            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm
            )
            self.hist_encoder_optimizer.step()

            mean_hist_latent_loss += hist_latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(
                p=2, dim=1
            ).mean()
            self.depth_encoder_optimizer.zero_grad()
            depth_encoder_loss.backward()
            nn.utils.clip_grad_norm_(
                self.depth_encoder.parameters(), self.max_grad_norm
            )
            self.depth_encoder_optimizer.step()
            return depth_encoder_loss.item()

    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):
        if self.if_depth:
            depth_actor_loss = (
                actions_teacher_batch.detach() - actions_student_batch
            ).norm(p=2, dim=1).mean()
            yaw_loss = (
                yaw_teacher_batch.detach() - yaw_student_batch
            ).norm(p=2, dim=1).mean()

            loss = depth_actor_loss + yaw_loss

            self.depth_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                self.max_grad_norm
            )
            self.depth_actor_optimizer.step()
            return depth_actor_loss.item(), yaw_loss.item()

    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):
        if self.if_depth:
            depth_encoder_loss = (
                scandots_latent_batch.detach() - depth_latent_batch
            ).norm(p=2, dim=1).mean()
            depth_actor_loss = (
                actions_teacher_batch.detach() - actions_student_batch
            ).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_(
                [*self.depth_actor.parameters(), *self.depth_encoder.parameters()],
                self.max_grad_norm
            )
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()

    def update_distill(self, actions_student_batch, actions_teacher_batch):
        if self.if_distill:
            loss = (actions_teacher_batch.detach() - actions_student_batch).norm(
                p=2, dim=1
            ).mean()

            self.student_actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.student_actor.parameters(), self.max_grad_norm
            )
            self.student_actor_optimizer.step()
            return loss.item()

    def update_counter(self):
        self.counter += 1

    def calc_amp_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self.amp_discriminator(amp_obs)
            # prob = 1 / (1 + torch.exp(-disc_logits)) 
            # disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))

            disc_r = torch.clamp(1 - (1/4) * torch.square(disc_logits - 1), min=0)

        return disc_r
    
    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward