import torch
import torch.nn as nn
from distributions import Categorical, DiagGaussian

class Policy(nn.Module):
    def __init__(self, net, action_space):
        super(Policy, self).__init__()

        assert isinstance(net, torch.nn.Module)
        self.net = net

        self.deterministic = False

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.net.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.net.output_size, num_outputs)
        else:
            raise NotImplementedError

    def set_deterministic(self, determ):
        self.deterministic = determ

    def act(self, inputs):
        critic_value, actor_value = self.net(inputs)

        m = torch.distributions.Categorical(actor_value)

        print("Actor_value: ", actor_value)

        dist = self.dist(actor_value)
        print("Dist: ", dist)

        if self.deterministic:
            #action = dist.mode()
            action = m.mode()
        else:
            action = dist.sample()
            action = m.sample()
        #print(dist.probs)
        action_log_probs = dist.log_probs(action)
        _ = dist.entropy().mean()

        return critic_value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self.net(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.net(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
