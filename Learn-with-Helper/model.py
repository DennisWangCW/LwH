from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import norm_col_init, weights_init, weights_init_mlp


class A3C_MLP(torch.nn.Module):
    def __init__(self, observation_space, action_space, n_frames):
        super(A3C_MLP, self).__init__()
        self.action_space = action_space

        self.training_steps = nn.Linear(1, 1)
        self.training_steps.weight.requires_grad = False
        self.training_steps.bias.requires_grad = False

        self.fc1 = nn.Linear(observation_space.shape[0], 256)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, action_space.shape[0])
        self.actor_linear2 = nn.Linear(128, action_space.shape[0])

        self.apply(weights_init_mlp)
        self.training_steps.weight.data = torch.Tensor([0])
        self.training_steps.bias.data = torch.Tensor([0])

        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.actor_linear2.weight.data = norm_col_init(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x = inputs

        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))
        return self.critic_linear(x), torch.Tensor([self.action_space.high[0]]) * F.softsign(self.actor_linear(x)), \
               0.5 * (F.softsign(self.actor_linear2(x)) + 1.0) + 1e-5
