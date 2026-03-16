from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch, torchvision

from legged_gym import LEGGED_GYM_ROOT_DIR, ASE_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.legged_robot_command import LeggedRobot_command, euler_from_quaternion
from legged_gym.utils.math import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from .lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch

# from rsl_rl.runners import OnPolicyRunnerMimic

import sys
sys.path.append(os.path.join(ASE_DIR, "ase"))
sys.path.append(os.path.join(ASE_DIR, "ase/utils"))
import cv2

from motion_lib import MotionLib
import torch_utils

class H1Command(LeggedRobot_command):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        # Simon: to save the obs demo when inferring
        self.obs_demo_save = []

        
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)

        # self.num_privileged_obs = self.cfg.env.priv_num_observations



        self.train_estimator = self.cfg.task.train_estimator

        # Pre init for motion loading
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        
        self.init_motions(cfg)

        BaseTask.__init__(self, self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.global_counter = 0
        self.total_env_steps_counter = 0

        # init low pass filter
        if self.cfg.control.action_filt:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.cfg.env.num_envs*self.cfg.env.num_actions),
                                                        highcut=np.ones(self.cfg.env.num_envs*self.cfg.env.num_actions) * self.cfg.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.cfg.env.num_envs * self.cfg.env.num_actions, 
                                                        device=self.device)
        # self.init_motion_buffers(cfg)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device = self.device)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos[:, :10])
        
        self.initialize_zmp()
        self.reset_idx(torch.arange(self.num_envs, device=self.device), init=True)
        self.post_physics_step()


    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        noise_scale_vec[:, :3] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, 3:5] = self.cfg.noise.noise_scales.imu
        if self.cfg.task.motion_task == 'walk':
            noise_scale_vec[:, 5:5+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
            noise_scale_vec[:, 5+self.num_dof:5+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
            noise_scale_vec[:, 5+3*self.num_dof:8+3*self.num_dof] = self.cfg.noise.noise_scales.gravity
        else:
            noise_scale_vec[:, 7:7+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
            noise_scale_vec[:, 7+self.num_dof:7+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        return noise_scale_vec
    
    def init_motions(self, cfg):
        self._key_body_ids = torch.tensor([3, 6, 9, 12], device=self.device)  #self._build_key_body_ids_tensor(key_bodies)

        self._key_body_ids_sim = torch.tensor([1, 4, 5, # Left Hip yaw, Knee, Ankle
                                               6, 9, 10,
                                               12, 15, 16, # Left Shoulder pitch, Elbow, hand
                                               17, 20, 21], device=self.device)
        
        # self._key_body_ids_sim_subset = torch.tensor([6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._key_body_ids_sim_subset = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        # self._key_body_ids_sim_subset = torch.tensor([0, 1, 3, 4, 6, 7, 8, 9, 10, 11], device=self.device)  # no knee and ankle
        self._num_key_bodies = len(self._key_body_ids_sim_subset)
        self._dof_body_ids = [1, 2, 3, # Hip, Knee, Ankle
                              4, 5, 6,
                              7,       # Torso
                              8, 9, 10, # Shoulder, Elbow, Hand
                              11, 12, 13]  # 13
        self._dof_offsets = [0, 3, 4, 5, 8, 9, 10, 
                             11, 
                             14, 15, 16, 19, 20, 21]  # 14

    

    def _load_motion(self, motion_file, no_keybody=False):
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device, 
                                     no_keybody=no_keybody, 
                                     regen_pkl=self.cfg.motion.regen_pkl)
        return

    def initialize_zmp(self):
        self.weighted_position_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.weighted_velocity_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.last_com_vel = torch.zeros(self.num_envs, 3 , device=self.device)


    def compute_zmp(self):
        total_mass = 0
        self.weighted_position_sum = torch.zeros(self.num_envs, 3 , device=self.device)
        self.weighted_velocity_sum = torch.zeros(self.num_envs, 3 , device=self.device)


        for i, body in enumerate(self.body_properties):
            # print(body)

            mass = body.mass
            # print(body.com)
            position = self.rigid_body_states[:, i, 0:3]
            # print(position)
            velocity = self.rigid_body_states[:, i, 7:10]
            # print(position.shape)

            self.weighted_position_sum[:, 0] += position[:, 0] * mass
            self.weighted_position_sum[:, 1] += position[:, 1] * mass
            self.weighted_position_sum[:, 2] += position[:, 2] * mass


            self.weighted_velocity_sum[:, 0] += velocity[:, 0] * mass
            self.weighted_velocity_sum[:, 1] += velocity[:, 1] * mass
            self.weighted_velocity_sum[:, 2] += velocity[:, 2] * mass

            total_mass +=mass

        # The position of the central mass
        com_pos = torch.cat(
            (self.weighted_position_sum[:, 0].view(-1,1) / total_mass,
            self.weighted_position_sum[:, 1].view(-1,1) / total_mass,
            self.weighted_position_sum[:, 2].view(-1,1) / total_mass), -1
        ).view(self.num_envs, 3)

        # The position of the central mass
        com_vel = torch.cat(
            (self.weighted_velocity_sum[:, 0].view(-1,1)  / total_mass,
            self.weighted_velocity_sum[:, 1].view(-1,1)  / total_mass,
            self.weighted_velocity_sum[:, 2].view(-1,1)  / total_mass), -1
        ).view(self.num_envs, 3)

        # print(self.contact_filt)


        # dt = self.cfg.sim.dt
        com_acc = (com_vel - self.last_com_vel) / self.dt


        self.zmp_x = com_pos[:,0] - (com_pos[:,2] / 9.81) * com_acc[:, 0]
        self.zmp_y = com_pos[:,1] - (com_pos[:,2] / 9.81) * com_acc[:, 1]


        self.last_com_vel = com_vel 



        # Step 1: Determine contact status for both feet in each environment
        # `contact_status` has shape [num_envs, 2] where each entry is True/False
        contact_status = self.contact_filt # Shape: [num_envs, 2]
        # print(contact_status)

        # Step 2: Get the (x, y) positions of both feet for each environment
        feet_xy = self.rigid_body_states[:, self.feet_indices, :2] # Shape: [num_envs, 2, 2]

        # Step 3: Initialize support center tensor for each environment
        support_center = torch.zeros((feet_xy.shape[0], 2), device=feet_xy.device) # Shape: [num_envs, 2]

        # Step 4: Calculate the support center based on contact conditions
        # Single support (left foot only)
        left_support_mask = (contact_status[:, 0]) & (~contact_status[:, 1]) # Shape: [num_envs]
        support_center[left_support_mask] = feet_xy[left_support_mask, 0, :]

        # Single support (right foot only)
        right_support_mask = (~contact_status[:, 0]) & (contact_status[:, 1])
        support_center[right_support_mask] = feet_xy[right_support_mask, 1, :]

        # Double support (both feet)
        double_support_mask = contact_status[:, 0] & contact_status[:, 1]
        support_center[double_support_mask] = (feet_xy[double_support_mask, 0, :] + feet_xy[double_support_mask, 1, :]) / 2.0

        # No contact mask
        no_contact_mask = ~(contact_status[:, 0] | contact_status[:, 1])

        # Step 5: Calculate the ZMP distance from the support center for each environment
        zmp_position = torch.stack((self.zmp_x, self.zmp_y), dim=-1) # Shape: [num_envs, 2]
        self.zmp_distance = torch.norm(zmp_position - support_center, dim=-1) # Euclidean distance for each environment
        self.zmp_distance[no_contact_mask] = 0.0

        # Output the ZMP distance for each environment
        # print("ZMP Distance from Support Center for each environment:", self.zmp_distance)

    
    def step(self, actions):
        actions = self.reindex(actions).to(self.device)
        actions.to(self.device)
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        # if self.global_counter % 10 ==0:
        #     actions[:,7] = 10
        #     actions[:,3] = 10
        # else:
        #     actions[:,7] = 10
        #     actions[:,3] = 10

        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)

        
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            # self.delay = torch.randint(0, 3, (1,), device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms


        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.global_counter += 1
        self.total_env_steps_counter += 1


        
        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        # clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)


        self.render()
                
        # self.actions[:, [4, 9]] = torch.clamp(self.actions[:, [4, 9]], -0.5, 0.5)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # for i in torch.topk(self.torques[self.lookat_id], 3).indices.tolist():
        #     print(self.dof_names[i], self.torques[self.lookat_id][i])
        
        self.post_physics_step()
        # print(self._in_place_flag)

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # def resample_motion_times(self, env_ids):
    #     return self._motion_lib.sample_time(self._motion_ids[env_ids])


    # def update_motion_ids(self, env_ids):
    #     self._motion_times[env_ids] = self.resample_motion_times(env_ids)
    #     self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
    #     self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])


    def domain_randomization(self, env_ids):
        if len(env_ids) == 0:
            return
            

        if self.cfg.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.cfg.env.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.cfg.env.num_actions), device=self.device)
    


    def reset_idx(self, env_ids, init=False):
        if len(env_ids) == 0:
            return
        
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)


   

        # reset robot states
        # self._reset_dofs(env_ids, None, None)
        self._reset_dofs(env_ids)

        # self._reset_root_states(env_ids, None, None, None)
        self._reset_root_states(env_ids)
        
        self._resample_commands(env_ids)  # no resample commands
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)



        self.domain_randomization(env_ids)


        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.

        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[:] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.  # reset obs history buffer TODO no 0s
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        # self.cur_goal_idx[env_ids] = 0
        # self.reach_goal_timer[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        # self.extras["episode"]["curriculum_completion"] = completion_rate_mean
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.episode_length_buf[env_ids] = 0


        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        return

   
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            if self.cfg.env.randomize_start_pos:
                self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            if self.cfg.env.randomize_start_yaw:
                rand_yaw = self.cfg.env.rand_yaw_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                if self.cfg.env.randomize_start_pitch:
                    rand_pitch = self.cfg.env.rand_pitch_range*torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
                else:
                    rand_pitch = torch.zeros(len(env_ids), device=self.device)
                quat = quat_from_euler_xyz(0*rand_yaw, rand_pitch, rand_yaw) 
                self.root_states[env_ids, 3:7] = quat[:, :]  
            if self.cfg.env.randomize_start_y:
                self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)

        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    

    def post_physics_step(self):
        super().post_physics_step()


        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            # self.draw_rigid_bodies_demo()
            # self.draw_rigid_bodies_actual()

        return

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()

    def _randomize_gravity(self, external_force = None):
        if self.cfg.domain_rand.randomize_gravity and external_force is None:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity


        sim_params = self.gym.get_sim_params(self.sim)
        gravity = external_force + torch.Tensor([0, 0, -9.81]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)

    def compute_obs_buf_commands(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        # print(self.commands[:3,:3])
        return torch.cat((#motion_id_one_hot,
                            self.base_ang_vel  * self.obs_scales.ang_vel,   #[1,3]
                            imu_obs,    #[1,2]
                            # torch.sin(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            # torch.cos(self.yaw - self.target_yaw)[:, None],  #[1,1]
                            # self.target_pos_rel,  
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            # self.reindex(self.action_history_buf[:, -1]),
                            self.last_actions,
                            self.projected_gravity
                            # self.reindex_feet(self.contact_filt.float()*0-0.5),
                            ),dim=-1)
    

    
    def compute_observations(self):
        # motion_id_one_hot = torch.zeros((self.num_envs, self._motion_lib.num_motions()), device=self.device)
        # motion_id_one_hot[torch.arange(self.num_envs, device=self.device), self._motion_ids] = 1.
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        diff = self.dof_pos[:, :10] - self.ref_dof_pos
        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        # self.zmp_distance
        self.compute_zmp()

        motion_features = self.obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)#self._demo_obs_buf[:, 2:, :].clone().flatten(start_dim=1) 
        priv_motion_features = self.priv_obs_history_buf[:, -self.cfg.env.prop_hist_len:].flatten(start_dim=1)

        # priv_explicit = torch.cat(( self.base_lin_vel * self.obs_scales.lin_vel,), dim=-1)
        priv_explicit = torch.cat(( self.base_lin_vel * self.obs_scales.lin_vel, self.zmp_distance.unsqueeze(1)), dim=-1)
        # priv_explicit = torch.cat(( self.base_lin_vel * self.obs_scales.lin_vel, diff), dim=-1)
        
        priv_latent = torch.cat((      # dim: 43
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.motor_strength[0] - 1, 
            self.motor_strength[1] - 1
        ), dim=-1)

        # print(self.friction_coeffs_tensor)

        obs_buf = self.compute_obs_buf_commands()
        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3]), dim=1)

        obs_buf = torch.cat((obs_buf, self.command_input), dim = -1)
        priv_obs_buf = torch.cat((obs_buf, priv_latent, diff, stance_mask, contact_mask), dim = -1)


        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec * self.cfg.noise.noise_scale


        if self.train_estimator == True:
            self.obs_buf = torch.cat([motion_features, obs_buf, priv_explicit], dim=-1)
        else:
            self.obs_buf = torch.cat([motion_features, obs_buf], dim=-1)

        self.privileged_obs_buf = torch.cat([priv_motion_features, priv_obs_buf, priv_explicit], dim=-1)
        # print(self.privileged_obs_buf.shape)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            # torch.stack([obs_buf] * 10, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.priv_obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
            # torch.stack([obs_buf] * 10, dim=1),
            torch.cat([
                self.priv_obs_history_buf[:, 1:],
                priv_obs_buf.unsqueeze(1)
            ], dim=1)
        )



        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # roll_cutoff = torch.abs(self.roll) > 1.0
        # pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5

        # print(self.roll, self.pitch)
        roll_cutoff = torch.abs(self.roll) > 1.0
        pitch_cutoff = torch.abs(self.pitch) > 1.0
        # height_cutoff = self.root_states[:, 2] < 0.5
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        # self.reset_buf |= height_cutoff

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        # self.ref_dof_pos = torch.zeros_like(self.dof_pos[:, :10])
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_l[torch.abs(sin_pos_l) < 0.1] = 0
        self.ref_dof_pos[:, 2] =  sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_hip_pitch_joint']
        self.ref_dof_pos[:, 3] =  sin_pos_l * scale_2 + self.cfg.init_state.default_joint_angles['left_knee_joint']
        self.ref_dof_pos[:, 4] =  sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_ankle_joint']
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        sin_pos_r[torch.abs(sin_pos_r) < 0.1] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r * scale_1 - self.cfg.init_state.default_joint_angles['right_hip_pitch_joint']
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2 - self.cfg.init_state.default_joint_angles['right_knee_joint']
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_1 - self.cfg.init_state.default_joint_angles['right_ankle_joint']
        # # Double support phase
        # self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos


    ######### Rewards #########
    def compute_reward(self):
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            
            name = self.reward_names[i]
            # print(name)
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew #if "demo" not in name else 0  # log demo rew but do not include in additative reward
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        if self.cfg.rewards.clip_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=-0.5)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        

    # def _reward_tracking_vx(self):
    #     rew = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
    #     # print('command', self.commands[:, 0])
    #     # print("vx rew", rew, self.base_lin_vel[:, 0], self.commands[:, 0])
    #     return rew
    
    # def _reward_tracking_ang_vel(self):
    #     rew = torch.minimum(self.base_ang_vel[:, 2], self.commands[:, 2]) / (self.commands[:, 2] + 1e-5)
    #     return rew
    

    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
    #     # print("lower dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 0])
    #     out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    #     # print("upper dof pos error: ", self.dof_pos - self.dof_pos_limits[:, 1])
    #     return torch.sum(out_of_limits, dim=1)
    
    # def _reward_stand_still(self):
    #     dof_pos_error = torch.norm((self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
    #     dof_vel_error = torch.norm(self.dof_vel[:, :11], dim=1)
    #     rew = torch.exp(- 0.1*dof_vel_error) * torch.exp(- dof_pos_error) 
    #     if self.cfg.task.motion_task == 'walk':
    #         rew *= torch.norm(self.commands[:, :2], dim=1) < 0.1 # only reward for zero command
    #     return rew
    

    # def _reward_feet_drag(self):
    #     # print(contact_bool)
    #     # contact_forces = self.contact_forces[:, self.feet_indices, 2]
    #     # print(contact_forces[self.lookat_id], self.force_sensor_tensor[self.lookat_id, :, 2])
    #     # print(self.contact_filt[self.lookat_id])
    #     feet_xyz_vel = torch.abs(self.rigid_body_states[:, self.feet_indices, 7:10]).sum(dim=-1)
    #     dragging_vel = self.contact_filt * feet_xyz_vel
    #     rew = dragging_vel.sum(dim=-1)
    #     # print(rew[self.lookat_id].cpu().numpy(), self.contact_filt[self.lookat_id].cpu().numpy(), feet_xy_vel[self.lookat_id].cpu().numpy())
    #     return rew
    
    # def _reward_energy(self):
    #     return torch.norm(torch.abs(self.torques * self.dof_vel), dim=-1)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     # stance_mask = self._get_gait_phase()
    #     # contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts) 
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     if self.cfg.task.motion_task == 'walk':
    #         rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     # rew_airTime[self._in_place_flag] = 0
    #     # print(self._in_place_flag)
    #     return rew_airTime

    # def _reward_feet_contact_number(self):
    #     """
    #     Calculates a reward based on the number of feet contacts aligning with the gait phase. 
    #     Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    #     """
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.
    #     stance_mask = self._get_gait_phase()
    #     reward = torch.where(contact == stance_mask, 1, -0.3)
    #     return torch.mean(reward, dim=1)


    # def _reward_feet_clearance(self):
    #     """
    #     Calculates reward based on the clearance of the swing leg from the ground during movement.
    #     Encourages appropriate lift of the feet during the swing phase of the gait.
    #     """
    #     # Compute feet contact mask
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.

    #     # Get the z-position of the feet and compute the change in z-position
    #     feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05
    #     delta_z = feet_z - self.last_feet_z
    #     self.feet_height += delta_z
    #     self.last_feet_z = feet_z

    #     # Compute swing mask
    #     swing_mask = 1 - self._get_gait_phase()

    #     # feet height should be closed to target feet height at the peak
    #     rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) > 0.01
    #     rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    #     self.feet_height *= ~contact
    #     return rew_pos

    # def _reward_joint_pos(self):
    #     """
    #     Calculates the reward based on the difference between the current joint positions and the target joint positions.
    #     """
    #     joint_pos = self.dof_pos[:, :10].clone()
    #     pos_target = self.ref_dof_pos.clone()
    #     diff = joint_pos - pos_target
    #     r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    #     return r


    # def _reward_feet_height(self):
    #     feet_height = self.rigid_body_states[:, self.feet_indices, 2]
    #     rew = torch.clamp(torch.norm(feet_height, dim=-1) - 0.2, max=0)
    #     # rew[self._in_place_flag] = 0
    #     if self.cfg.task.motion_task == 'walk':
    #         rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     # print("height: ", rew[self.lookat_id])
    #     return rew
    
    # def _reward_feet_force(self):
    #     rew = torch.norm(self.contact_forces[:, self.feet_indices, 2], dim=-1)
    #     rew[rew < 500] = 0
    #     rew[rew > 500] -= 500
    #     # rew[self._in_place_flag] = 0
    #     # print(rew[self.lookat_id])
    #     # print(self.dof_names)
    #     return rew

    # # def _reward_dof_error(self):
    # #     # dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
    # #     if self.cfg.task.motion_task == 'walk':    
    # #         # loss_upper = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, 11:], dim=1) * 0.7
    # #         # loss_down = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1) * 0.3
    # #         # loss_upper = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, 11:], dim=1) * 2.0
    # #         # loss_down = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1)
    # #         # loss_upper = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, 10:], dim=1) * 2.0
    # #         # loss_down = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :10], dim=1)
    # #         # dof_error = loss_upper + loss_down
    # #         dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    # #     else:
    # #         dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, :11], dim=1)

    # #     return dof_error

    # def _reward_arms_dof_error(self):  
    #     dof_error = torch.sum(torch.square(self.dof_pos[:, 11:] - self.default_dof_pos[:, 11:]), dim=1)
    #     return dof_error

    # def _reward_waist_dof_error(self):  
    #     dof_error = torch.square(self.dof_pos[:, 10] - self.default_dof_pos[:, 10])
    #     return dof_error
    
    # def _reward_hip_yaw_dof_error(self):
    #     dof_error = torch.square(self.dof_pos[:,0] + self.dof_pos[:,5]  - self.default_dof_pos[:,0] - self.default_dof_pos[:,5])
    #     return dof_error

    # def _reward_hip_roll_dof_error(self):
    #     dof_error = torch.square(self.dof_pos[:,1] + self.dof_pos[:,6]  - self.default_dof_pos[:,1] - self.default_dof_pos[:,6])
    #     return dof_error

    # def _reward_tracking_lin_vel_commands(self):
    #     rew = torch.exp(- torch.sum(torch.square(self.base_lin_vel[:, :2] - self.commands[:, :2]), dim=1) * self.cfg.rewards.tracking_sigma)
    #     # rew[self._in_place_flag] = 0
    #     # rew = torch.exp(- 4 * torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1))
    #     return rew

    # def _reward_tracking_ang_vel_commands(self):
    #     rew = torch.exp(- torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2]) * self.cfg.rewards.tracking_sigma)
    #     # rew[self._in_place_flag] = 0
    #     # rew = torch.exp(- 4 * torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 2]))
    #     return rew

    # def _reward_base_height(self):
    #     base_height = self.root_states[:, 2]
    #     rew = torch.square(base_height - self.cfg.rewards.base_height_target)
    #     # rew[self._in_place_flag] = 0
    #     return rew

    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel), dim=1)

    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques), dim=1)

    # def _reward_single_feet_contact(self): # add
    #     # penalize feet not touching the ground and both feet touching the ground
    #     left_feet_contact = torch.where(self.contact_forces[:, self.feet_indices[0], 2] > 1, torch.ones_like(self.contact_forces[:, 0, 0]), torch.zeros_like(self.contact_forces[:, 0, 0]))
    #     right_feet_contact = torch.where(self.contact_forces[:, self.feet_indices[1], 2] > 1, torch.ones_like(self.contact_forces[:, 0, 0]), torch.zeros_like(self.contact_forces[:, 0, 0]))
    #     # No contact for both feet
    #     rew_both_no_contact = (1 - left_feet_contact) * (1 - right_feet_contact)  
    #     # Both feet in contact
    #     rew_both_contact = left_feet_contact * right_feet_contact 
    #     # Single foot contact
    #     rew = torch.ones_like(right_feet_contact) - rew_both_no_contact - rew_both_contact
    #     if self.cfg.task.motion_task == 'walk':
    #         rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     # Return both single foot contact reward and penalties for both no contact and both contact
    #     return rew

    # # def _reward_feet_away(self):
    # #     # Reward for keeping feet at a reasonable distance
    # #     left_foot = self.rigid_body_states[:, self.feet_indices[0], :3]
    # #     right_foot = self.rigid_body_states[:, self.feet_indices[1], :3]
    # #     foot_distance = torch.norm(left_foot - right_foot, dim=1)
    # #     # 10.29     0.40    0.38     0.35
    # #     rew = torch.minimum(foot_distance, torch.tensor(0.40, device=self.device))
    # #     return rew

    # # def _reward_knee_away(self):
    # #     # Reward for keeping feet at a reasonable distance
    # #     left_knee = self.rigid_body_states[:, self.knee_indices[0], :3]
    # #     right_knee = self.rigid_body_states[:, self.knee_indices[1], :3]
    # #     knee_distance = torch.norm(left_knee - right_knee, dim=1)
    # #     # May be 0.4 too high?  0.40   0.38    0.35
    # #     rew = torch.minimum(knee_distance, torch.tensor(0.40, device=self.device))
    # #     return rew


    # def _reward_knee_away(self):
    #     """
    #     Reward for keeping knees within a reasonable distance range.
    #     Rewards maximum if within [min_dist, max_dist] range, 
    #     penalizes when exceeding max_dist or below min_dist.
    #     """
    #     left_knee = self.rigid_body_states[:, self.knee_indices[0], :3]
    #     right_knee = self.rigid_body_states[:, self.knee_indices[1], :3]
    #     knee_distance = torch.norm(left_knee - right_knee, dim=1)

    #     # Define minimum and maximum desired distances
    #     min_dist = torch.tensor(0.30, device=self.device)  # Minimum desirable distance
    #     max_dist = torch.tensor(0.50, device=self.device)  # Maximum desirable distance

    #     # Calculate reward based on how well the knee distance stays within the range
    #     # Reward is full when within [min_dist, max_dist], penalizes if outside this range
    #     within_range = (knee_distance >= min_dist) & (knee_distance <= max_dist)
    #     reward_within = knee_distance * within_range.float()

    #     # Penalize deviations below min_dist or above max_dist
    #     penalty_below = torch.exp(-torch.abs(knee_distance - min_dist) * 100) * (knee_distance < min_dist)
    #     penalty_above = torch.exp(-torch.abs(knee_distance - max_dist) * 100) * (knee_distance > max_dist)

    #     # Combine reward and penalties
    #     reward = reward_within + penalty_below + penalty_above
    #     return reward


    # def _reward_feet_away(self):
    #     """
    #     Reward for keeping feet within a reasonable distance range.
    #     Rewards maximum if within [min_dist, max_dist] range,
    #     penalizes when exceeding max_dist or below min_dist.
    #     """
    #     left_foot = self.rigid_body_states[:, self.feet_indices[0], :3]
    #     right_foot = self.rigid_body_states[:, self.feet_indices[1], :3]
    #     foot_distance = torch.norm(left_foot - right_foot, dim=1)

    #     # Define minimum and maximum desired distances
    #     min_dist = torch.tensor(0.30, device=self.device)  # Minimum desirable distance
    #     max_dist = torch.tensor(0.50, device=self.device)  # Maximum desirable distance

    #     # Calculate reward for distances within the range [min_dist, max_dist]
    #     within_range = (foot_distance >= min_dist) & (foot_distance <= max_dist)
    #     reward_within = foot_distance * within_range.float()

    #     # Penalize deviations below min_dist or above max_dist
    #     penalty_below = torch.exp(-torch.abs(foot_distance - min_dist) * 100) * (foot_distance < min_dist)
    #     penalty_above = torch.exp(-torch.abs(foot_distance - max_dist) * 100) * (foot_distance > max_dist)

    #     # Combine reward and penalties
    #     reward = reward_within + penalty_below + penalty_above
    #     return reward






    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos[:, :10].clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_body_states[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_body_states[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        foot_speed_norm = torch.norm(self.rigid_body_states[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)    

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        # self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - 1000.0).clip(0, 400), dim=1)

    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        # joint_diff = self.dof_pos - self.default_dof_pos
        joint_diff = self.dof_pos[:, :10] - self.default_dof_pos[:, :10]
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 5: 7]    # default [6 : 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    
    def _reward_upper_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        shoulder_roll_diff = self.dof_pos[:, 12] + self.dof_pos[:, 16] - self.default_dof_pos[:, 12] - self.default_dof_pos[:, 16]
        shoulder_yaw_diff = self.dof_pos[:, 13] + self.dof_pos[:, 17] - self.default_dof_pos[:, 13] - self.default_dof_pos[:, 17]
        torso_diff = self.dof_pos[:, 10] - self.default_dof_pos[:, 10]
        # joint_diff = self.dof_pos[:, 10:] - self.default_dof_pos[:, 10:]
        return - 0.04 * torch.abs(torso_diff) - 0.04 * torch.abs(shoulder_roll_diff) - 0.04 * torch.abs(shoulder_yaw_diff)


    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(
            self.rigid_body_states[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_body_states[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_alive(self):
        return 0.1
    # def _reward_action_smoothness(self):
    #     """
    #     Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
    #     This is important for achieving fluid motion and reducing mechanical stress.
    #     """
    #     term_1 = torch.sum(torch.square(
    #         self.action_history_buf[:, -1] - self.actions), dim=1)
    #     term_2 = torch.sum(torch.square(
    #         self.actions + self.action_history_buf[:, -2] - 2 * self.action_history_buf[:, -1]), dim=1)
    #     term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
    #     return term_1 + term_2 + term_3

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3


#####################################################################
###=========================jit functions=========================###
#####################################################################

# # @torch.jit.script
# def build_demo_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, local_key_body_pos, dof_offsets):
#     local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
#     local_root_vel = quat_rotate_inverse(root_rot, root_vel)
#     # print('root_vel',root_vel)
#     # print('local_root_vel', local_root_vel)
#     # print(local_root_vel[0])

#     # heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
#     # local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)
#     # local_root_vel = quat_rotate(heading_rot, root_vel)
#     # print(local_root_vel[0], "\n")

#     # root_pos_expand = root_pos.unsqueeze(-2)  # [num_envs, 1, 3]
#     # local_key_body_pos = key_body_pos - root_pos_expand
    
#     # heading_rot_expand = heading_rot.unsqueeze(-2)
#     # heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
#     # flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
#     # flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
#     # local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
#     # flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
#     roll, pitch, yaw = euler_from_quaternion(root_rot)
#     return torch.cat((dof_pos, local_root_vel, local_root_ang_vel, roll[:, None], pitch[:, None], root_pos[:, 2:3], local_key_body_pos.view(local_key_body_pos.shape[0], -1)), dim=-1)

# @torch.jit.script
# def reindex_motion_dof(dof, indices_sim, indices_motion, valid_dof_body_ids):
#     dof = dof.clone()
#     dof[:, indices_sim] = dof[:, indices_motion]
#     return dof[:, valid_dof_body_ids]

@torch.jit.script
def local_to_global(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = rigid_body_pos.reshape(total_bodies, 3)
    global_body_pos = quat_rotate(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3) + root_pos[:, None, :3]
    return global_body_pos

@torch.jit.script
def global_to_local(quat, rigid_body_pos, root_pos):
    num_key_bodies = rigid_body_pos.shape[1]
    num_envs = rigid_body_pos.shape[0]
    total_bodies = num_key_bodies * num_envs
    heading_rot_expand = quat.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, num_key_bodies, 1))
    flat_heading_rot = heading_rot_expand.view(total_bodies, heading_rot_expand.shape[-1])

    flat_end_pos = (rigid_body_pos - root_pos[:, None, :3]).view(total_bodies, 3)
    local_end_pos = quat_rotate_inverse(flat_heading_rot, flat_end_pos).view(num_envs, num_key_bodies, 3)
    return local_end_pos

@torch.jit.script
def global_to_local_xy(yaw, global_pos_delta):
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rotation_matrices = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=2).view(-1, 2, 2)
    local_pos_delta = torch.bmm(rotation_matrices, global_pos_delta.unsqueeze(-1))
    return local_pos_delta.squeeze(-1)

