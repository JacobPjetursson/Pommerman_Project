import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, net):
        super(Policy, self).__init__()
        assert isinstance(net, torch.nn.Module)
        self.net = net

    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        value, actor_features = self.net(inputs, masks)
        dist = torch.distributions.Categorical(probs=actor_features)

        if deterministic:
            action = dist.probs.argmax(dim=1, keepdim=True)  # mode
        else:
            action = dist.sample().unsqueeze(-1)

        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)

        _ = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.net(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action):
        value, actor_features = self.net(inputs, masks)
        dist = torch.distributions.Categorical(probs=actor_features)

        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy



