'''An agent that preforms a random action each step'''
from models.model_pomm import PommNet
from models.policy import Policy
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
from our_ppo import PPO
import math
import torch
import random
import pommerman


class PytorchAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, ppo_in, *args, **kwargs):
        super(PytorchAgent, self).__init__(*args, **kwargs)
        # self.nn = PommNet(torch.Size([4, 11, 11]))
        # self.policy = Policy(self.nn, action_space=Discrete(6))
        # self.ppo = PPO(self.policy, 2.5e-3)
        # self.ppo.set_deterministic(False)
        self.ppo = ppo_in

        self.rewards = []
        self.states = []
        self.critic_values = []
        self.actions = []
        self.action_log_probs = []
        self.zero = torch.zeros([1], dtype=torch.long)
        self.zero = self.zero.cuda()
        self.still = self.zero
        self.up = self.zero + 1
        self.down = self.zero + 2
        self.left = self.zero + 3
        self.right = self.zero + 4
        self.bomb = self.zero + 5
        self.pos_actions = [self.still, self.up, self.down, self.left, self.right, self.bomb]

    def get_rewards(self, state, persons, reward, l):
        steps = state["step_count"] / 8000
        persons_left = persons.nonzero().size(0)
        t_reward = reward + steps
        if reward < 0:
            t_reward = t_reward * persons_left
        return [float(t_reward) * math.pow(0.95, i) for i in range(l)]

    def model_step(self, state, reward):
        if state["step_count"] % 50 == 0:
            print(".", end="")

        features, _, persons, _ = self.getFeatures(state)
        self.states.append(features)

        if reward != 0 or state["step_count"] >= 800:  # len(self.actions) == self.number_of_actions or reward == -1:
            self.rewards = self.get_rewards(state, persons, reward, len(self.actions))
            self.ppo.update(self.states, self.critic_values, self.rewards, self.actions, self.action_log_probs)
            self.states = []
            self.critic_values = []
            self.actions = []
            self.rewards = []
            self.action_log_probs = []

    def getFeatures(self, obs):
        me = obs["position"]
        obs_persons = torch.FloatTensor((obs["board"] >= 9).astype(int))
        obs_persons[me[0], me[1]] -= 1

        # Board with me
        obs_board_me = torch.zeros(11, 11)
        obs_board_me[me[0], me[1]] += 1

        # Board with walls rigid + wood
        obs_board_walls = torch.FloatTensor((obs["board"] == pommerman.constants.Item.Rigid).astype(int))
        obs_board_walls += 0.5 * torch.FloatTensor((obs["board"] == pommerman.constants.Item.Wood).astype(int))

        # Board with flames
        obs_board_flames = torch.FloatTensor((obs["board"] == pommerman.constants.Item.Flames).astype(int))

        # Getting the blastmap
        obs_board_bombs_life = torch.FloatTensor((obs["bomb_life"]).astype(int))
        obs_board_bombs_blast = torch.FloatTensor((obs["bomb_blast_strength"]).astype(int))
        obs_blast_map = self.get_blast_map(obs_board_bombs_blast, obs_board_bombs_life)

        board = torch.cat([obs_board_me, obs_persons, obs_board_walls, obs_board_flames, obs_blast_map])

        return board, me, obs_blast_map, obs["ammo"]

    def get_blast_map(self, blast_strength, blast_life):
        blast_map = torch.zeros(11, 11)
        for y in range(11):
            for x in range(11):
                st = blast_strength[y, x]
                st = int(st.cpu().numpy())
                if st > 0:
                    y_start = max(y - (st - 1), 0)
                    y_end = min(y + st, 11)
                    x_start = max(x - (st - 1), 0)
                    x_end = min(x + st, 11)
                    blast_map[y_start:y_end, x] = blast_life[y, x]
                    blast_map[y, x_start:x_end] = blast_life[y, x]
        return blast_map

    def act(self, obs, action_space):
        features, me, blast_map, ammo = self.getFeatures(obs)
        critic_value, action, action_log_probs = self.ppo.policy.act(inputs=features)
        self.critic_values.append(critic_value)
        self.action_log_probs.append(action_log_probs)
        # val_act = self.get_valid_action(blast_map, me, ammo, action)
        self.actions.append(action)
        return action.cpu().numpy()[0]
