import os
import types

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from helpers.monitor import Monitor
from helpers.vec_env import VecEnvWrapper
from helpers.vec_env.dummy_vec_env import DummyVecEnv
from helpers.vec_env.subproc_vec_env import SubprocVecEnv
from helpers.vec_env.vec_normalize import VecNormalize
import envs.pommerman


def make_env(env_id, seed, rank, log_dir=None, allow_early_resets=False):
    def _thunk():
        env = envs.pommerman.make_env(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            env = Monitor(env, os.path.join(log_dir, str(rank)),
                          allow_early_resets=allow_early_resets)

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, no_norm, num_stack,
                  log_dir=None, device='cpu', allow_early_resets=False, eval=False):

    envs = [make_env(env_name, seed, i, log_dir, allow_early_resets)
                for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    if not no_norm and len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)
        if eval:
            # An ugly hack to remove updates
            def _obfilt(self, obs):
                if self.ob_rms:
                    obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                                  -self.clipob, self.clipob)
                    return obs
                else:
                    return obs
            envs._obfilt = types.MethodType(_obfilt, envs)

    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        return obs, reward, done, info
