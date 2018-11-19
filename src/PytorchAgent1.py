'''An agent that preforms a random action each step'''
from models.model_pomm import PommNet
from models.policy import Policy
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
from our_ppo import PPO
import math
import torch
import random


class PytorchAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(PytorchAgent, self).__init__(*args, **kwargs)
        self.nn = PommNet(torch.Size([4, 11, 11]))
        self.policy = Policy(self.nn, action_space=Discrete(6))
        self.ppo = PPO(self.policy, 2.5e-3)
        self.ppo.set_deterministic(False)

        self.rewards = []
        self.states = []
        self.critic_values = []
        self.actions = []
        self.action_log_probs = []
        self.number_of_actions = 4
        self.zero = torch.zeros([1], dtype=torch.long)
        self.zero = self.zero.cuda()
        self.ammo = 0

    def get_rewards(self, state, reward, l):
        t_reward = reward
        return [float(t_reward) * math.pow(0.95, i) for i in range(l)]

    def model_step(self, state, reward):
        features, _, _, _, _ = self.getFeatures(state)
        self.states.append(features)

        if reward != 0:  # len(self.actions) == self.number_of_actions or reward == -1:
            self.rewards = self.get_rewards(state, reward, len(self.actions))
            self.ppo.update(self.states, self.critic_values, self.rewards, self.actions, self.action_log_probs)
            self.states = []
            self.critic_values = []
            self.actions = []
            self.rewards = []
            self.action_log_probs = []

    def getFeatures(self, obs):
        me = obs["position"]
        obs_persons = torch.FloatTensor((obs["board"] == 10).astype(int)) + torch.FloatTensor(
            (obs["board"] == 11).astype(int)) + torch.FloatTensor(
            (obs["board"] == 12).astype(int))
        obs_board_walls = torch.FloatTensor((obs["board"] == 1).astype(int)) + torch.FloatTensor(
            (obs["board"] == 2).astype(int))
        obs_board_me = torch.FloatTensor((obs["board"] == 13).astype(int))
        obs_board_flames = torch.FloatTensor((obs["board"] == 4).astype(int))
        obs_board_bombs_life = torch.FloatTensor((obs["bomb_life"]).astype(int))
        obs_board_bombs = torch.FloatTensor((obs["bomb_life"] > 0).astype(int))
        obs_board_bombs_blast = torch.FloatTensor((obs["bomb_blast_strength"]).astype(int))
        obs_blast_map = self.get_blast_map(obs_board_bombs_blast, obs_board_bombs_life)
        board = torch.cat(
            [obs_board_walls + obs_board_bombs + obs_persons + obs_board_flames, obs_board_me, obs_blast_map,
             obs_persons])
        return board, me, obs_board_walls + obs_board_bombs + obs_persons + obs_board_flames, obs_blast_map, obs["ammo"]

    def get_blast_map(self, blast_strength, blast_life):
        blast_map = torch.zeros(11, 11)
        for y in range(11):
            for x in range(11):
                st = blast_strength[y, x]
                st = int(st.cpu().numpy())
                if st > 0 and blast_life[y, x] < 4:
                    st = st - 1
                    y_start = max(y - st, 0)
                    y_end = min(y + st, 11)
                    x_start = max(x - st, 0)
                    x_end = min(x + st, 11)
                    blast_map[y_start:y_end, x] = blast_life[y, x]
                    blast_map[y, x_start:x_end] = blast_life[y, x]
        return blast_map

    def can_go_further(self, obstacles, y, x):
        val_list = []
        if y > 0 and self.get_empty_pos(obstacles, y - 1, x):
            val_list.append(self.zero + 1)
        if y < 10 and self.get_empty_pos(obstacles, y + 1, x):
            val_list.append(self.zero + 2)
        if x > 0 and self.get_empty_pos(obstacles, y, x - 1):
            val_list.append(self.zero + 3)
        if x < 10 and self.get_empty_pos(obstacles, y, x + 1):
            val_list.append(self.zero + 4)
        return val_list

    def get_empty_pos(self, obstacles, y, x):
        tot = 0
        if y > 0 and not obstacles[y - 1, x]:
            tot += 1
        if y < 10 and not obstacles[y + 1, x]:
            tot += 1
        if x > 0 and not obstacles[y, x - 1]:
            tot += 1
        if x < 10 and not obstacles[y, x + 1]:
            tot += 1
        return tot > 0

    def get_valid_actions(self, walls_and_bombs, blastmap, me, ammo):
        valid_actions = []
        if me[0] > 0 and walls_and_bombs[me[0] - 1, me[1]] == 0 and (
                blastmap[me[0] - 1, me[1]] < 1 or blastmap[me[0] - 1, me[1]] > 2):
            valid_actions.append(self.zero + 1)
        if me[0] < 10 and walls_and_bombs[me[0] + 1, me[1]] == 0 and (
                blastmap[me[0] + 1, me[1]] < 1 or blastmap[me[0] + 1, me[1]] > 2):
            valid_actions.append(self.zero + 2)
        if me[1] > 0 and walls_and_bombs[me[0], me[1] - 1] == 0 and (
                blastmap[me[0], me[1] - 1] < 1 or blastmap[me[0], me[1] - 1] > 2):
            valid_actions.append(self.zero + 3)
        if me[1] < 10 and walls_and_bombs[me[0], me[1] + 1] == 0 and (
                blastmap[me[0], me[1] + 1] < 1 or blastmap[me[0], me[1] + 1] > 2):
            valid_actions.append(self.zero + 4)

        if not valid_actions and 0 < blastmap[me[0], me[1]] < 3:
            return [self.zero + 1, self.zero + 2, self.zero + 3, self.zero + 4]

        if valid_actions and ammo == self.ammo:
            valid_actions.append(self.zero + 5)
        valid_actions.append(self.zero)

        if len(self.actions) > 0 and self.actions[-1] == (self.zero + 5):
            bm_1 = (blastmap == 1).type('torch.FloatTensor')
            vas = self.can_go_further(walls_and_bombs + bm_1, me[0], me[1])
            return vas

        return valid_actions

    def get_valid_action(self, walls_and_bombs, blastmap, me, ammo, action):
        valid_actions = self.get_valid_actions(walls_and_bombs, blastmap, me, ammo)
        if action in valid_actions:
            return action
        elif not valid_actions or (self.zero in valid_actions and len(valid_actions) == 1):
            return self.zero
        else:
            if self.zero in valid_actions:
                valid_actions.remove(self.zero)
            return random.choice(valid_actions)

    def act(self, obs, action_space):
        features, me, walls_and_bombs, blastmap, ammo = self.getFeatures(obs)
        if ammo > self.ammo:
            self.ammo = ammo
        critic_value, action, action_log_probs = self.policy.act(inputs=features)
        self.critic_values.append(critic_value)
        self.action_log_probs.append(action_log_probs)
        val_act = self.get_valid_action(walls_and_bombs, blastmap, me, ammo, action)
        self.actions.append(val_act)
        return val_act.cpu().numpy()[0]
