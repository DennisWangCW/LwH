from policy_domain import Policy_Domain
from environment import create_env
import numpy as np
import json
import torch
import argparse
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--env', default='uav-v0', metavar='ENV', help='environment to train on (default: UAV)')
parser.add_argument('--max-episode-length', type=int, default=2000, metavar='M',
                    help='maximum length of an episode (default: 2000)')
parser.add_argument('--variance', type=float, nargs='+', metavar='V', help='variance of actions')
parser.add_argument('--demo-type', type=str, default='uav',
                    help='demonstration type, available choices are (uav, uav_wrong, else)')
parser.add_argument('--num-demos', default=500, type=int)
parser.add_argument('--prior-decay', default=0.0005, type=float)

if __name__ == '__main__':
    args = parser.parse_args()
    nb_demos = args.num_demos
    episode_length = args.max_episode_length
    seed = args.seed
    # demonstrations = {}
    env_id = args.env
    env = create_env(env_id, seed)

    actions_traj = []
    states_traj = []
    next_states_traj = []
    rewards_traj = []
    dones_traj = []
    returns_traj = []

    policy = Policy_Domain(env.observation_space, env.action_space)
    counter = 0
    for i in range(nb_demos):
        print("generation trajectory", i)
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        returns = 0
        obs = env.reset()
        for i in range(episode_length):
            # print("iteration", i)
            states.append(obs)
            # action = env.action_space.sample()
            mu, sigma = policy.forward(Variable(torch.Tensor(obs)), time_step=1, args=args)
            eps = torch.randn(mu.size())
            eps = Variable(eps)
            action = (mu + sigma.sqrt() * eps).data
            action = action.cpu().numpy()
            # print("action", action)
            obs, reward, done, info = env.step(action)
            # env.render()
            next_states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            returns += reward
            if done:
                counter += 1
                print("counter={}".format(counter))
                obs = env.reset()

        actions_traj.append(np.array(actions))
        states_traj.append(np.array(states))
        next_states_traj.append(np.array(next_states))
        rewards_traj.append(np.array(rewards))
        dones_traj.append(np.array(dones))
        returns_traj.append(np.array([returns]))

    name = args.demo_type + '_' + str(args.variance[0]).replace('.', '_')
    np.savez('./' + name + '.npz', acs=actions_traj, obs=states_traj, rews=rewards_traj, next_obs=next_states_traj,
             dones=dones_traj, ep_rets=returns_traj)
    #
    # load_data = np.load('./data.npz')
    # print("load", load_data.files)
    # print("shape", np.shape(load_data['obs']))
    #
    #     print("is generating episode", i)
    #
    #     observation_sequence = []
    #     action_sequence = []
    #     obs = env.reset()
    #     observation_sequence.append(np.expand_dims(np.copy(obs), 0))
    #     for _ in range(episode_length):
    #         action = policy.action_sample(torch.Tensor(obs), 1, args)
    #         obs, reward, done, info = env.step(action.data.cpu().numpy())
    #         observation_sequence.append(np.expand_dims(np.copy(obs), 0))
    #         action_sequence.append(np.expand_dims(np.copy(action.data.cpu().numpy()), 0))
    #         if done:
    #             break
    #     traj = {}
    #     traj['states'] = np.concatenate(observation_sequence)
    #     traj['actions'] = np.concatenate(action_sequence)
    #     demonstrations['traj'+str(i)] = traj
    #
    # name = 'demonstrations_' + str(args.variance[0]).replace('.', '_') + '_' + args.demo_type + '.npy'
    # print("name", name)
    # np.save(name, demonstrations)
    #
    # # np_load_old = np.load
    # # np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # # reads = np.load('demonstrations.npy').item()
    # # for k in reads.keys():
    # #     print(k)
    # # np.load = np_load_old



