import torch
from ddpg import DDPG
import numpy as np


class Config:
    def __init__(self):
        self.mode = 'train'
        self.env = 'UAV'
        self.hidden1 = 400
        self.hidden2 = 300
        self.ou_sigma = 0.2
        self.ou_mu = 0.0
        self.ou_theta = 0.15
        self.save_trained_models = 'trained_models/'
        self.epsilon = 50000
        self.seed = -1
        self.init_w = 0.003


class Policy_Domain:
    def __init__(self, observation_space, action_space):
        self.config = Config()
        self.agent_ddpg = DDPG(observation_space, action_space, self.config)
        if not (self.config.env == 'UAV'):
            self.agent_ddpg.load_weights(self.config.save_trained_models)
            self.agent_ddpg.eval()

        self.current_direct_wrong = 'north'
        self.min_distance_x = 50.0
        self.min_distance_y = 50.0

    def forward(self, state, time_step, args, reset_flag=False):
        if args.demo_type == 'uav':
            if args:
                coefs = args.variance * 2
                prior_decay = args.prior_decay
            else:
                coefs = [0.09, 0.09]
                prior_decay = 0.005
            time_step = torch.Tensor([time_step])[0]
            perspective = torch.atan(state[12] / state[13])
            first_perspective = torch.where(state[13] > 0,
                                            torch.where(state[12] > 0, perspective / np.pi * 180.0,
                                                        (perspective + 2 * np.pi) / np.pi * 180.0),
                                            (perspective + np.pi) / np.pi * 180.0)

            target = torch.atan(state[10] / state[11])
            position_target = torch.where(state[11] > 0,
                                          torch.where(state[10] > 0, target / np.pi * 180.0,
                                                      (target + 2 * np.pi) / np.pi * 180.0),
                                          (target + np.pi) / np.pi * 180.0)

            first_target = torch.remainder(first_perspective - position_target, 360.0)

            average_direction = torch.where(
                torch.sign(180.0 - first_target) + 1.0 > 0, -first_target / 180.0, (360.0 - first_target) / 180.0)
            variance_direction = 0.0 * average_direction + coefs[0]  # 0.1

            turning_free = torch.where(
                torch.sign(4 - torch.argmin(state[0:9]).float()) + 1.0 > 0, 45.0 + 0 * average_direction,
                -45.0 + 0 * average_direction)
            average_free = turning_free / 180.0
            variance_free = 0.0 * average_free + coefs[0]  # 0.1

            average_steer = torch.where(
                torch.sign(100 * torch.min(state[0:9]) - 15.0) + 1.0 > 0, average_direction, average_free)
            variance_steer = torch.where(
                torch.sign(100 * torch.min(state[0:9]) - 15.0) + 1.0 > 0, variance_direction, variance_free)

            speed = state[14]
            average_throttle = torch.clamp(2.5 - 50 * (speed / 2 + 0.5), -0.5, 0.5)

            variance_throttle = 0.0 * average_throttle + coefs[1]
            decay = prior_decay * (time_step - 1) + 1

            covariance = torch.cat(
                (variance_steer.unsqueeze_(0), variance_throttle.unsqueeze_(0)), 0) * decay

            average = torch.cat((average_steer.unsqueeze_(0), average_throttle.unsqueeze_(0)), 0)

        elif args.demo_type == 'uav_wrong':
            if reset_flag:
                self.current_direct_wrong = 'north'
                self.min_distance_x = 50.0
                self.min_distance_y = 50.0

            if args:
                coefs = args.variance * 2
                prior_decay = args.prior_decay
            else:
                coefs = [0.09, 0.09]
                prior_decay = 0.005
            time_step = torch.Tensor([time_step])[0]
            perspective = torch.atan(state[12] / state[13])
            first_perspective = torch.where(state[13] > 0,
                                            torch.where(state[12] > 0, perspective / np.pi * 180.0,
                                                        (perspective + 2 * np.pi) / np.pi * 180.0),
                                            (perspective + np.pi) / np.pi * 180.0)

            target = torch.atan(state[10] / state[11])
            position_target = torch.where(state[11] > 0,
                                          torch.where(state[10] > 0, target / np.pi * 180.0,
                                                      (target + 2 * np.pi) / np.pi * 180.0),
                                          (target + np.pi) / np.pi * 180.0)

            distance = (state[9] / 2 + 0.5) * (torch.sqrt(torch.Tensor([2])[0]) * 3000)

            distance_y = torch.abs(distance * torch.sin(2 * position_target / 360 * torch.Tensor([np.pi])[0]))
            distance_x = torch.abs(distance * torch.cos(2 * position_target / 360 * torch.Tensor([np.pi])[0]))

            if distance_y > self.min_distance_y:
                self.current_direct_wrong = 'north'
            elif distance_x > self.min_distance_x:
                if self.current_direct_wrong == 'north':
                    self.min_distance_x -= 5
                self.current_direct_wrong = 'east'
            else:
                if self.current_direct_wrong == 'east':
                    self.min_distance_y -= 5
                self.current_direct_wrong = 'north'

            if self.current_direct_wrong == 'north':
                if position_target > 0 and position_target < 180:
                    position_target = 90
                else:
                    position_target = 270

            else:
                if position_target < 90 or position_target > 270:
                    position_target = 0
                else:
                    position_target = 180

            first_target = torch.remainder(first_perspective - position_target, 360.0)

            average_direction = torch.where(
                torch.sign(180.0 - first_target) + 1.0 > 0, -first_target / 180.0, (360.0 - first_target) / 180.0)
            variance_direction = 0.0 * average_direction + coefs[0]  # 0.1

            turning_free = torch.where(
                torch.sign(4 - torch.argmin(state[0:9]).float()) + 1.0 > 0, 45.0 + 0 * average_direction,
                -45.0 + 0 * average_direction)

            average_free = turning_free / 180.0
            variance_free = 0.0 * average_free + coefs[0]  # 0.1
            average_steer = torch.where(
                torch.sign(100 * torch.min(state[0:9]) - 15.0) + 1.0 > 0, average_direction, average_free)
            variance_steer = torch.where(
                torch.sign(100 * torch.min(state[0:9]) - 15.0) + 1.0 > 0, variance_direction, variance_free)

            speed = state[14]
            average_throttle = torch.clamp(2.5 - 50 * (speed / 2 + 0.5), -0.5, 0.5)
            variance_throttle = 0.0 * average_throttle + coefs[1]

            decay = prior_decay * (time_step - 1) + 1

            covariance = torch.cat(
                (variance_steer.unsqueeze_(0), variance_throttle.unsqueeze_(0)), 0) * decay
            average = torch.cat((average_steer.unsqueeze_(0), average_throttle.unsqueeze_(0)), 0)
        else:
            average = self.agent_ddpg.select_action(state)
            time_step = torch.Tensor([time_step])[0]
            decay = args.prior_decay * (time_step - 1) + 1
            covariance = torch.ones(average.shape) * 0.1 * decay

        return average, covariance

    def action_sample(self, state, time_step, args):
        average, covariance = self.forward(state, time_step, args)
        eps = torch.Tensor(np.random.normal(0, 1, average.shape))
        action = average + eps * covariance.sqrt()
        return action
