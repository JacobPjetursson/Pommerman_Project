import torch
import torch.nn as nn
from distributions import Categorical, DiagGaussian

class Policy(nn.Module):
    def __init__(self, nn, action_space):
        super(Policy, self).__init__()

        assert isinstance(nn, torch.nn.Module)
        self.nn = nn

        self.deterministic = False

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.nn.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.nn.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.dist = self.dist.cuda()

    def set_deterministic(self, determ):
        self.deterministic = determ

    @property
    def is_recurrent(self):
        return self.nn.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.nn.recurrent_hidden_state_size

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs):
        critic_value, actor_value = self.nn(inputs)
        dist = self.dist(actor_value)

        if self.deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        #_ = dist.entropy().mean()

        return critic_value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self.nn(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.nn(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
