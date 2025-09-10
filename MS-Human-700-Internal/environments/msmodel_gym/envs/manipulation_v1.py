import os
from typing import Dict, Any, Optional
import numpy as np
from gymnasium.utils import EzPickle
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import mujoco
from msmodel_gym.envs.utils import get_render_fps, action_obs_check, euler2quat

class ManipulationEnvV1(MujocoEnv, EzPickle):

    metadata: Dict[str, Any] = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "w_pos": 50, "k_pos": 10.0,
        "w_ori": 10, "k_ori": 2.0,
        "w_reach": 10, "k_reach": 10.0,
        "w_lift": 1, "k_lift": 0.02,
        "w_act": 1, 
        "w_drop": 100.0, "k_drop": 0.3,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        skip_frames: int = 5,
        reset_noise_scale = 1e-3,
        target_pos_range = None,
        target_ori_range = None,
        reward_dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs
    ):

        model_path = os.path.dirname(__file__) + "/../../../models/MS-Human-700-Manipulation.xml"
        model_path = os.path.abspath(model_path)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        fps = get_render_fps(model_path, skip_frames)
        self.metadata["render_fps"] = fps
        self.control_timestep = 1 / fps

        EzPickle.__init__(
            self,
            render_mode,
            skip_frames,
            reset_noise_scale,
            **kwargs
        )

        self.render_mode = render_mode
        self._reset_noise_scale = reset_noise_scale
        self.target_pos_range = target_pos_range
        self.target_ori_range = target_ori_range
        self.target_pos = np.array([0, 0, 0.3])
        self.target_ori = np.array([0, 0])   # Rx and Ry

        self.pos_threshold = 0.02
        self.ori_threshold = 0.1

        self.reward_weight = reward_dict

        # Initialize MujocoEnv
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        MujocoEnv.__init__(
            self, model_path, skip_frames, observation_space=observation_space, render_mode=render_mode, camera_name="record_camera", **kwargs
        )

        self.init_qpos[:] = self.model.key_qpos[0].copy()
        self.body_name_list = [self.model.body(body_id).name for body_id in range(self.model.nbody)]
        self.joint_name_list = [self.model.joint(jnt_id).name for jnt_id in range(self.model.njnt)]
        self.muscle_name_list = [self.model.actuator(act_id).name for act_id in range(self.model.nu)]

        self.object_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "Object")
        self.target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.success_indicator_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_indicator")
        self.hand_body_list = ['proximal_thumb', 'distal_thumb', 
                               '2proxph', '2midph', '2distph',
                               '3proxph', '3midph', '3distph',
                               '4proxph', '4midph', '4distph',
                               '5proxph', '5midph', '5distph']
        self.obj_init_xpos = self.data.site_xpos[self.object_sid].copy()
        
        observation, _ = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation.shape[0],), dtype=np.float64)
        action_obs_check(self)

        print("observation space shape: ", self.observation_space.shape)
        print("action space shape: ", self.action_space.shape)
    
    def seed(self, seed=0):
        pass
            
    @property
    def terminated(self):
        terminated = self._get_done()
        return terminated
    
    def _get_obs(self):
        qpos = self.data.qpos.flat.copy() 
        qvel = self.data.qvel.flat.copy() 
        qacc = self.data.qacc.flat.copy()

        act = self.data.act.flat.copy()
        actuator_forces = self.data.actuator_force.flat.copy() / 1000
        actuator_forces = actuator_forces.clip(-100, 100)
        actuator_length = self.data.actuator_length.flat.copy()
        actuator_velocity = self.data.actuator_velocity.flat.copy().clip(-100, 100)
        
        sim_time = np.array([self.data.time])
        self.obj_xpos = self.data.site_xpos[self.object_sid].copy()
        self.target_xpos = self.target_pos + self.obj_init_xpos
        self.obj_xpos_dist = self.target_xpos - self.obj_xpos
        self.obj_ori = self.data.qpos[-3:-1].copy()
        self.obj_ori_dist = self.target_ori - self.obj_ori

        reach_dist = 0
        for body_name in self.hand_body_list:
            body_xpos = self.data.body(body_name).xpos.copy()
            reach_dist += np.linalg.norm(body_xpos - self.obj_xpos)
        self.reach_dist = np.abs(reach_dist) / len(self.hand_body_list)

        obs_dict = {
            "qpos": qpos,
            "qvel": qvel,
            "qacc": qacc,
            "act": act,
            "actuator_forces": actuator_forces,
            "actuator_length": actuator_length,
            "actuator_velocity": actuator_velocity,
            "sim_time": sim_time,
            "obj_xpos": self.obj_xpos,
            "target_xpos": self.target_xpos,
            "obj_ori": self.obj_ori,
            "target_ori": self.target_ori,
            "obj_xpos_dist": self.obj_xpos_dist,
            "obj_ori_dist": self.obj_ori_dist,
            "reach_dist": np.array([self.reach_dist]),
        }
        
        observation = np.concatenate([obs_dict[key] for key in obs_dict.keys()])

        return observation, obs_dict
    
    def step(self, action):
        
        self.do_simulation(action, self.frame_skip)

        observation, obs_dict = self._get_obs()

        pos_reward = self.reward_weight["w_pos"] * np.exp(-self.reward_weight["k_pos"] * np.linalg.norm(self.obj_xpos_dist))
        ori_reward = self.reward_weight["w_ori"] * np.exp(-self.reward_weight["k_ori"] * np.linalg.norm(self.obj_ori_dist))
        reach_reward = self.reward_weight["w_reach"] * np.exp(-self.reward_weight["k_reach"] * self.reach_dist)
        drop = self.reach_dist > self.reward_weight["k_drop"]

        lift_bonus = (self.obj_xpos[2] - self.obj_init_xpos[2] >= self.reward_weight["k_lift"]) and (not drop)
        lift_reward = self.reward_weight["w_lift"] * float(lift_bonus)

        act_penalty = -self.reward_weight["w_act"] * np.linalg.norm(action) / self.model.na

        done = self._get_done()
        done_penalty = -self.reward_weight["w_drop"] * float(done)
        
        reward = pos_reward * ori_reward + reach_reward + lift_reward + act_penalty + done_penalty

        solved = np.linalg.norm(self.obj_xpos_dist) < self.pos_threshold \
                and np.linalg.norm(self.obj_ori_dist) < self.ori_threshold\
                and (not drop)
        self.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if solved else np.array([2, 0])
        self.model.site_size[self.success_indicator_sid, :] = np.array([0.1,]) if solved else np.array([0.001,])
        
        # Terminate and truncate condition
        terminated = self.terminated
        truncated = False

        info = {
            "reward": reward,
            "reward_pos": pos_reward,
            "reward_ori": ori_reward,
            "reward_reach": reach_reward,
            "reward_lift": lift_reward,
            "reward_act": act_penalty,
            "reward_done": done_penalty,
            "solved": solved,
        }

        return observation, reward, terminated, truncated, info
    
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        self.obj_init_xpos = self.data.site_xpos[self.object_sid].copy()

        if self.target_pos_range is not None:
            self.target_pos = np.random.uniform(low=self.target_pos_range[0], high=self.target_pos_range[1])
        if self.target_ori_range is not None:
            self.target_ori = np.random.uniform(low=self.target_ori_range[0], high=self.target_ori_range[1])

        target_euler = np.array([self.target_ori[0], self.target_ori[1], 0])
        self.model.body_pos[self.target_bid] = self.target_pos + self.obj_init_xpos
        self.model.body_quat[self.target_bid] = euler2quat(target_euler)

        observation, obs_dict = self._get_obs()
        
        return observation
    
    def render(self, mode=None):
        return super().render()
    
    def _get_done(self):
        
        x, y, z = self.data.site_xpos[self.object_sid].copy()
        if self.reach_dist > self.reward_weight["k_drop"]:
            return True
        if x < 0 or x > 0.7:
            return True
        if y < -0.5 or y > 0.5:
            return True
        if z < (self.obj_init_xpos[2] - 0.02) or z > 1.7:
            return True
        
        return False
