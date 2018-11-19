'''An agent that preforms a random action each step'''
from models.model_pomm import PommNet
from models.policy import Policy
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
from our_ppo import PPO
import math
import torch


class PytorchAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(PytorchAgent, self).__init__(*args, **kwargs)
        self.nn = PommNet(torch.Size([3, 11, 11]))
        self.policy = Policy(self.nn, action_space=Discrete(6))
        self.ppo = PPO(self.policy, 5e-4)
        self.ppo.set_deterministic(False)

        self.rewards = []
        self.states = []
        self.critic_values = []
        self.actions = []
        self.action_log_probs = []
        self.number_of_actions = 8
        self.zero = torch.zeros([1], dtype=torch.long)
        self.zero = self.zero.cuda()

    def get_number_of_zeros(self):
#        return self.actions.count(self.zero);
        count = 0
        for i in list(reversed(self.actions[:-1])):
            if i == self.zero:
                count = count + 1
            else:
                break
        return count

    def get_rewards(self, state, reward, l):
        t_reward = reward
        t_reward = t_reward - self.get_number_of_zeros() * 1
        rews = [float(t_reward) * math.pow(0.95, i) for i in range(l)]
        return rews

    def model_step(self, state, reward):
        self.states.append(self.getFeatures(state))

        if len(self.actions) == self.number_of_actions or reward == -1:
            self.rewards = self.get_rewards(state, reward, len(self.actions))
            self.ppo.update(self.states, self.critic_values, self.rewards, self.actions, self.action_log_probs)
            self.states = []
            self.critic_values = []
            self.actions = []
            self.rewards = []
            self.action_log_probs = []

    def getFeatures(self, obs):
        obs_board = torch.FloatTensor(obs["board"]) / 16
        obs_bomb_blast = torch.FloatTensor(obs["bomb_blast_strength"])
        obs_bomb_life = torch.FloatTensor((obs["bomb_life"] > 0).astype(int))
        board = torch.cat([obs_board, obs_bomb_life, obs_bomb_blast])
        return board

    def act(self, obs, action_space):
        features = self.getFeatures(obs)
        critic_value, action, action_log_probs = self.policy.act(inputs=features)
        self.critic_values.append(critic_value)
        self.action_log_probs.append(action_log_probs)
        self.actions.append(action)
        return action.cpu().numpy()[0]
