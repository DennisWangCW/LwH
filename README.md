# LwH
This repository includes codes for the algorithm Deep Reinforcement Learning with Non-Expert Helper (LwH) proposed in the paper titled "Deep Reinforcement Learning-based Autonomous UAV Navigation with Sparse Rewards"

Before running the code, you need to install the environment 'uav-v0' (which can be found at) or other gym environments with continuous action psaces and low-dimensional observation space (e.g., mujoco envs).

To run the code, simply execute: python main.py --env=uav-v0 --variance=0.4 --demo-type=uav --use-prior
  where 'variance' specifies the initial variance of the prior policy, 'demo-type' specifies the prior policy type (uav->SenAvo-Pri;  uav_wrong->Naive-Pri)

To evaluate the learned policy, simply execute: python gym_eval.py --env=uav-v0 --variance=0 --load-model-dir=path-to-the-model
