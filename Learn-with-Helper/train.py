from __future__ import division
from setproctitle import setproctitle as ptitle
from utils import setup_logger
import logging
import numpy as np
import torch
import torch.optim as optim
from environment import create_env
from utils import ensure_shared_grads
from model import A3C_MLP
from player_util import Agent
from torch.autograd import Variable
import os


def train(rank, args, shared_model, optimizer):
    init = True
    ptitle('Training Agent: {}'.format(rank))
    torch.manual_seed(args.seed + rank)
    env = create_env(args.env, args.seed + rank)
    # env = gym.make(args.env)
    # env.seed(args.seed + rank)

    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    player = Agent(None, env, args, None, rank)
    player.model = A3C_MLP(
        player.env.observation_space, player.env.action_space, args.stack_frames)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.model.train()

    if rank == 1:
        file = open(os.path.join(args.log_dir, 'TD_Error.txt'), 'w+')

    local_step_counter = 0
    while True:
        if init:
            shared_model.training_steps.weight.data \
                .copy_(torch.Tensor([0]))
            shared_model.training_steps.bias.data \
                .copy_(torch.Tensor([0]))
            init = False
        player.model.load_state_dict(shared_model.state_dict())   
        for step in range(args.num_steps):
            # print("thread", rank, local_step_counter, shared_model.training_steps.weight.data.cpu().numpy())
            local_step_counter += 1
            shared_model.training_steps.weight.data \
                .copy_(torch.Tensor([1]) + shared_model.training_steps.weight.data)

            player.action_train()
            if player.done:
                break

        terminal = False
        if player.done or player.eps_len >= args.max_episode_length:
            terminal = True

        R = torch.zeros(1)
        if not player.done:
            state = player.state
            value, _, _ = player.model(Variable(state))
            R = value.data

        if terminal:
            shared_model.training_steps.bias.data \
                .copy_(torch.Tensor([1]) + shared_model.training_steps.bias.data)
            player.eps_len = 0
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            player.reset_flag = True

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + np.float(player.rewards[i])
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            if rank == 1:
                file.write(str(advantage.pow(2).data.cpu().numpy()[0]))
                file.write(' ')
                file.write(str(int(shared_model.training_steps.weight.data.cpu().numpy()[0])))
                file.write('\n')

            player.values[i] = player.values[i].float()
            player.values[i+1] = player.values[i+1].float()
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                (player.log_probs[i].sum() * Variable(gae)) - \
                (0.01 * player.entropies[i].sum())

        player.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model)
        optimizer.step()
        player.clear_actions()
        if shared_model.training_steps.weight.data.cpu().numpy() > args.training_steps:
            break
