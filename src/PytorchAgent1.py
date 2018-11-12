'''An agent that preforms a random action each step'''
from models.model_pomm import PommNet
from models.policy import Policy
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
from our_ppo import PPO
import torch


class PytorchAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(PytorchAgent, self).__init__(*args, **kwargs)
        self.nn = PommNet(torch.Size([3, 11, 11]))
        self.policy = Policy(self.nn, action_space=Discrete(6))
        self.ppo = PPO(self.policy, 2.5e-4)
        self.ppo.set_deterministic(False)

        self.rewards = []
        self.states = []
        self.critic_values = []
        self.actions = []
        self.action_log_probs = []

    def model_step(self, state, reward):
        self.states.append(self.getFeatures(state))
        self.rewards.append(reward)

        if len(self.actions) == 8:
            self.ppo.update(self.states, self.critic_values, self.rewards, self.actions, self.action_log_probs)
            self.states = []
            self.critic_values = []
            self.actions = []
            self.rewards = []
            self.action_log_probs = []

    def getFeatures(self, obs):
        board = torch.FloatTensor([obs["board"], obs["bomb_blast_strength"], obs["bomb_life"]]) / 15
        return board

    def act(self, obs, action_space):
        features = self.getFeatures(obs)
        critic_value, action, action_log_probs = self.policy.act(inputs=features)
        self.critic_values.append(critic_value)
        self.action_log_probs.append(action_log_probs)
        self.actions.append(action)
        return action.cpu().numpy()[0]
