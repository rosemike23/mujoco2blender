import os
from warnings import warn

import numpy as np
from gymnasium import spaces

try:
    import mujoco
except ImportError:
    warn("MuJoCo not found. Please install MuJoCo from https://www.mujoco.org/downloads/.")

def action_obs_check(cls):
    low = cls.action_space.low
    high = cls.action_space.high
    if (low == high).any():
        raise ValueError("Action space has the same low and high value")

    low = cls.observation_space.low
    high = cls.observation_space.high
    if (low == high).any():
        raise ValueError("Observation space has the same low and high value")

def get_observation_space(xml_path, get_obs_fn, obs_kwargs=None):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    obs = get_obs_fn(data, **obs_kwargs if obs_kwargs is not None else {})
    assert obs.ndim == 1, "Observation must be 1D"

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float64)
    return observation_space

def get_render_fps(xml_path, skip_frames):
    model = mujoco.MjModel.from_xml_path(xml_path)
    timestep = model.opt.timestep

    return int(round(1.0 / timestep / skip_frames))

def euler2quat(euler):
    """ Convert Euler Angles to Quaternions """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat

def joint_name_to_dof_index(all_joint_name_list, joint_name_list):
    """
    Convert joint name list to joint index list
    """
    joint_index_list = []
    for joint_name in joint_name_list:
        if joint_name in all_joint_name_list:
            joint_index_list.append(all_joint_name_list.index(joint_name))
        else:
            raise ValueError("Joint name {} not found in all joint name list".format(joint_name))
    return joint_index_list