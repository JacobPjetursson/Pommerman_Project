import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_generic import NNBase
import numpy as np


class ConvNet4(nn.Module):
    def __init__(self, input_shape, num_channels=64, output_size=512,
                 batch_norm=True, activation_fn=F.relu, dilation=False):
        super(ConvNet4, self).__init__()

        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.activation_fn = activation_fn
        self.flattened_size = num_channels * (input_shape[1] - 4) * (input_shape[2] - 4)
        self.drop_prob = 0.2

        self.conv1 = nn.Conv2d(input_shape[0], num_channels, 3, stride=(1, 1), padding=1)
        if dilation:
            self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, dilation=2, padding=2)
        else:
            self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(num_channels)
            self.bn2 = nn.BatchNorm2d(num_channels)
            self.bn3 = nn.BatchNorm2d(num_channels)
            self.bn4 = nn.BatchNorm2d(num_channels)

        else:
            self.bn1 = lambda x: x
            self.bn2 = lambda x: x
            self.bn3 = lambda x: x
            self.bn4 = lambda x: x

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation_fn(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation_fn(x)

        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = self.activation_fn(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        out = self.fc2(x)

        return out


class PommNet(NNBase):
    def __init__(self, obs_shape, recurrent=False, hidden_size=512, batch_norm=True):
        super(PommNet, self).__init__(recurrent, hidden_size, hidden_size)
        self.obs_shape = obs_shape

        # FIXME hacky, recover input shape from flattened observation space
        # assuming an 11x11 board and 3 non spatial features
        bs = 11
        self.other_shape = [3]
        input_channels = (obs_shape[0] - self.other_shape[0]) // (bs*bs)
        self.image_shape = [3, bs, bs]
        assert np.prod(obs_shape) >= np.prod(self.image_shape)

        self.common_conv = ConvNet4(
            input_shape=self.image_shape,
            output_size=hidden_size,
            batch_norm=batch_norm)

        self.actor = nn.Linear(hidden_size, hidden_size)

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, inputs):
        x_test = inputs.reshape((-1, 3, 11, 11))
        # inputs_image = inputs[:, :-self.other_shape[0]].view([-1] + self.image_shape)
        #       print(inputs_image.size)
        # inputs_other = inputs[:, -self.other_shape[0]:]

        x = self.common_conv(x_test)
        # x = x_conv + x_mlp

        out_actor = self.actor(x)
        out_value = self.critic(x)

        return out_value, out_actor
