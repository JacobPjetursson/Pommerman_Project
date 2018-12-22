import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, net, action_space):
        super(Policy, self).__init__()
        assert isinstance(net, torch.nn.Module)
        self.net = net


    @property
    def is_recurrent(self):
        return self.net.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.net.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.net(inputs, rnn_hxs, masks)
        dist = torch.distributions.Categorical(probs=actor_features)

        if deterministic:
            action = dist.probs.argmax(dim=1, keepdim=True) # mode
        else:
            action = dist.sample().unsqueeze(-1)

        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)

        _ = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.net(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.net(inputs, rnn_hxs, masks)
        dist = torch.distributions.Categorical(probs=actor_features)

        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



