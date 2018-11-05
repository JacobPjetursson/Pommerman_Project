'''An agent that preforms a random action each step'''
from models.model_pomm import PommNet
from models.policy import Policy
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
import torch


class PytorchAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(PytorchAgent, self).__init__(*args, **kwargs)
        self.nn = PommNet(torch.Size([3, 11, 11]))
        self.policy = Policy(self.nn, action_space=Discrete(6))

    def step(self, reward):
        return

    def getFeatures(self, obs):
        board = torch.FloatTensor([obs["board"], obs["bomb_blast_strength"], obs["bomb_life"]])
        return board

    def act(self, obs, action_space):
        features = self.getFeatures(obs)
        value, action, action_log_probs = self.policy.act(inputs=features)
        return action.numpy()[0]
