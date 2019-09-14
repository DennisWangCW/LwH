from __future__ import division
from setproctitle import setproctitle as ptitle
import numpy as np
import torch
from environment import create_env
from utils import setup_logger
from model import A3C_MLP
from player_util import Agent
import time
import logging
import queue


class Model_Buffer:
    def __init__(self, args):
        self.args = args
        self.model_buffer = queue.Queue(-1)
        self.flag = -1

    def put(self, model):
        training_steps = model.training_steps.weight.data.cpu().numpy()[0].astype(np.int)
        training_episodes = model.training_steps.bias.data.cpu().numpy()[0].astype(np.int)
        flag = np.int(training_steps / self.args.cache_interval)
        if flag > self.flag:
            self.flag = flag
            self.model_buffer.put([model.state_dict(),
                                   np.int(flag*self.args.cache_interval), np.int(training_episodes)])
        return self

    def get(self):
        if self.model_buffer.empty():
            return False
        else:
            return self.model_buffer.get()

    def get_flag(self):
        return self.flag

    def qsize(self):
        return self.model_buffer.qsize()

    def clear(self):
        self.model_buffer = queue.Queue(-1)
        self.flag = -1


def test(rank, args, shared_model):
    model_buffer = Model_Buffer(args)
    test_episodes = args.test_episodes
    ptitle('Test Agent')
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    print("logfile check", r'{0}{1}_log'.format(args.log_dir, args.env))

    print("logs in test", args.log_dir)

    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    # for i in range(100):
    #     log['{}_log'.format(args.env)].info('{0}'.format(i))

    # print('we prefix seed = -1 when testing')
    # args.seed = -1
    torch.manual_seed(args.seed)
    env = create_env(args.env, args.seed)
    # env = gym.make(args.env)
    # env.seed(args.seed)

    start_time = time.time()
    num_tests = 0
    player = Agent(None, env, args, None, rank)
    player.model = A3C_MLP(
        player.env.observation_space, player.env.action_space, args.stack_frames)
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.done = True

    player.model.eval()

    is_model_empty = True
    is_testing = False
    while True:
        model_buffer.put(shared_model)
        if player.done and np.mod(num_tests, test_episodes) == 0 and not is_testing:
            reward_episode = 0
            success_rate = 0
            load_model = model_buffer.get()
            model_queue_size = model_buffer.qsize()
            if load_model:
                is_testing = True
                is_model_empty = False
                training_steps = load_model[1]
                training_episodes = load_model[2]
                player.model.load_state_dict(load_model[0])
            else:
                is_model_empty = True
                time.sleep(10)

        if not is_model_empty:
            player.action_test()
            # log['{}_log'.format(args.env)].info("test steps {}".format(1))
            reward_episode += player.reward
            if 'is_success' in player.info.keys():
                success_rate += 1
                # print("crash detected")

            if player.done:
                eps_len_temp = player.eps_len

                num_tests += 1
                player.eps_len = 0
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()

                if np.mod(num_tests, test_episodes) == 0:
                    is_testing = False
                    reward_episode = reward_episode / test_episodes
                    success_rate = success_rate / test_episodes
                    log['{}_log'.format(args.env)].info(
                        "Time {0}, training episodes {1}, training steps {2}, reward episode {3}, success_rate {4}, "
                        "model cached {5}"
                            .format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                                    training_episodes, training_steps, reward_episode, success_rate,
                                    model_queue_size))

                    # save model:
                    state_to_save = player.model.state_dict()
                    # torch.save(state_to_save, '{0}{1}.dat'.format(args.save_model_dir, args.env))
                    # torch.save(state_to_save, '{0}{1}_pre.dat'.format(args.save_model_dir, args.env))

                    torch.save(state_to_save, '{0}{1}.dat'.format(args.log_dir, args.env))
                    torch.save(state_to_save, '{0}{1}_pre.dat'.format(args.log_dir, args.env))
        if training_steps > args.training_steps:
            break
