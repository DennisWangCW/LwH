from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import create_env
from model import A3C_MLP
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time
import datetime
import os.path as osp
from utils import str_process
import subprocess
import gym
import gym_uav


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--env',
    default='uav-v0',
    metavar='ENV',
    help='environment to train on (default: UAV)')
parser.add_argument(
    '--workers',
    type=int,
    default=16,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 300)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=2000,
    metavar='M',
    help='maximum length of an episode (default: 2000)')

parser.add_argument(
    '--variance',
    type=float,
    nargs='+',
    metavar='V',
    help='variance of actions')

parser.add_argument(
    '--prior-decay',
    type=float,
    default=0.0005,
    metavar='PD',
    help='decay slope of prior (default: 0.0005)')

parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load',
    default=False,
    metavar='L',
    help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir',
    default='logs',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--model',
    default='MLP',
    metavar='M',
    help='Model type to use')
parser.add_argument(
    '--stack-frames',
    type=int,
    default=1,
    metavar='SF',
    help='Choose number of observations to stack')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--training-steps',
    type=int,
    default=int(2E7))

parser.add_argument(
    '--l2-regular',
    type=int,
    default=1e-5)

parser.add_argument(
    '--test-episodes',
    type=int,
    default=50,
    metavar='TE',
    help='number of episodes used for testing the trained policy (default: 100)')

parser.add_argument(
    '--cache-interval',
    type=int,
    default=50000,
    metavar='CI',
    help='each interval, cache model into queue (default: 10000)')

parser.add_argument('--use-prior', default=False, action='store_true')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    args.seed = args.seed * 1201
    torch.manual_seed(args.seed)
    env = create_env(args.env, -1)

    shared_model = A3C_MLP(env.observation_space, env.action_space, args.stack_frames)
    shared_model.share_memory()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.load:
        saved_state = torch.load('{0}{1}.dat'.format(
            args.load_model_dir, args.env), map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)

    # optimizer for policy and value optimization
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    # reinforcement learning with non-expert helper
    processes = []
    p = mp.Process(target=test, args=(-1, args, shared_model))
    p.start()
    processes.append(p)
    time.sleep(0.1)
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(
            rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.1)
        
    for p in processes:
        time.sleep(0.1)
        p.join()
