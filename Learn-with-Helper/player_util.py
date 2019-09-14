from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import Variable
from utils import normal
from policy_domain import Policy_Domain


class Agent(object):
    def __init__(self, model, env, args, state, rank):
        self.time_step = 0
        self.model = model
        self.env = env
        self.state = state
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.infos = []
        self.entropies = []
        self.done = True
        self.reward = 0
        self.info = None
        self.rank = rank
        self.action_pre = []
        self.action_pre_sup = []
        self.reset_flag = False
        self.action_test_collection = []

        self.prior = Policy_Domain(env.observation_space, env.action_space)

    def action_train(self):
        self.time_step += 1
        value, mu_learned, sigma_learned = self.model(Variable(self.state))

        if self.args.use_prior:
            mu_prior, sigma_prior = self.prior.forward(Variable(self.state), self.time_step, self.args)
            sigma_prior = sigma_prior.diag()

        sigma_learned = sigma_learned.diag()

        self.reset_flag = False

        if self.args.use_prior:
            sigma = (sigma_learned.inverse() + sigma_prior.inverse()).inverse()
            temp = torch.matmul(sigma_learned.inverse(), mu_learned) + torch.matmul(sigma_prior.inverse(), mu_prior)
            mu = torch.matmul(sigma, temp)
        else:
            sigma = sigma_learned
            mu = mu_learned
        
        sigma = sigma.diag()
        sigma_learned = sigma_learned.diag()

        eps = torch.randn(mu.size())
        pi = np.array([math.pi])
        pi = torch.from_numpy(pi).float()
        eps = Variable(eps)
        pi = Variable(pi)

        action = (mu + sigma.sqrt() * eps).data

        act = Variable(action)
        prob = normal(act, mu, sigma)
        action = torch.clamp(action, self.env.action_space.low[0], self.env.action_space.high[0])
        entropy = 0.5 * ((sigma_learned * 2 * pi.expand_as(sigma_learned)).log() + 1)
        self.entropies.append(entropy)
        log_prob = (prob + 1e-6).log()
        self.log_probs.append(log_prob)
        state, reward, self.done, self.info = self.env.step(action.cpu().numpy())

        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done

        self.values.append(value)
        self.rewards.append(reward)
        self.infos.append(self.info)
        return self

    def action_test(self):
        with torch.no_grad():
            value, mu, sigma = self.model(Variable(self.state))

        action = mu.data.cpu().numpy()
        self.action_test_collection.append(action)
        state, self.reward, self.done, self.info = self.env.step(action)

        self.state = torch.from_numpy(state).float()
        self.eps_len += 1
        self.done = self.done or self.eps_len >= self.args.max_episode_length
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.action_pre = []
        self.action_pre_sup = []
        return self
