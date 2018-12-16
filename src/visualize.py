import argparse
import os
import types

import numpy as np
import torch
from helpers.vec_env.vec_normalize import VecNormalize
from models.factory import create_policy
from envs import make_vec_envs


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PommeFFACompetitionFast-v0',
                    help='environment to train on (default: PommeFFACompetitionFast-v0)')
parser.add_argument('--load-path', default='trained_models/a2c/PommeFFACompetitionFast-v0.pt',
                    help='path to checkpoint file')
parser.add_argument('--recurrent-policy', action='store_true', default=False,
                    help='use a recurrent policy')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--no-norm', action='store_true', default=True,
                    help='disables normalization')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA')
parser.add_argument('--use-search', action='store_true', default = False)
parser.add_argument('--hide', action='store_true', default = False)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

###### Set to true if we want to show with the 
use_search = args.use_search

num_env = 1
env = make_vec_envs(args.env_name, args.seed + 1000,
                    num_env, gamma=None, no_norm=args.no_norm,
                    num_stack=args.num_stack, log_dir=None, add_timestep=args.add_timestep,
                    device=device, eval=True, allow_early_resets=False, use_search=use_search)

# Get a render function
render_func = None
tmp_env = env
while True:
    if hasattr(tmp_env, 'envs'):
        render_func = tmp_env.envs[0].render
        break
    elif hasattr(tmp_env, 'venv'):
        tmp_env = tmp_env.venv
    elif hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env
    else:
        break

# We need to use the same statistics for normalization as used in training
state_dict, ob_rms = torch.load(args.load_path)

# FIXME this is very specific to Pommerman env right now
actor_critic = create_policy(
    env.observation_space,
    env.action_space,
    name='pomm',
    nn_kwargs={
        #'conv': 'conv3',
        'batch_norm': True,
        'recurrent': args.recurrent_policy,
        'hidden_size': 512,
    },
    train=False)

actor_critic.load_state_dict(state_dict)
actor_critic.to(device)

recurrent_hidden_states = torch.zeros(num_env, actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(num_env, 1).to(device)

obs = env.reset()

if args.hide:
    render_func = None

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if p.getBodyInfo(i)[0].decode() == "torso":
            torsoId = i

rewards = []
wins = 0
deaths = 0

step = 0

while True:
    step = step + 1
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)

    obs, reward, done, _ = env.step(action)

    masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    for i, d in enumerate(done):
        if d:
            rewards.append(reward[i].item())
            if reward[i].item() > 0:
                wins = wins + 1
            if reward[i].item() < 0 and step <= 800:
                deaths = deaths + 1
            print("Game ended in {} steps, total games played: {}. Win rate: {}. Survival rate {}".format(step-1, len(rewards), float(wins) / len(rewards), 1.0-float(deaths)/len(rewards)))
            step = 0

    if render_func is not None:
        render_func('human')

