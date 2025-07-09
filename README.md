# finsearch-25-DeepRL-F70
This repo conatins the Inverted pendulum problem solved using DQN Algorithm
# Key Components
DQN Agent with:

Online and target Q-networks

Experience Replay Buffer

Epsilon-greedy exploration

Discretization of continuous actions to enable DQN

Training loop with evaluation after each episode

Plots for evaluation scores and learning curves

# . DQN with Action Discretization (Our Implementation)
In this version, we discretize the continuous action space (e.g., from -2 to +2) into a fixed number of torque levels (21 here).

A standard Deep Q-Network (DQN) agent is trained on this modified environment.

While it shows learning progress, this method does not fully solve the environment — i.e., the agent struggles to consistently keep the pendulum upright.

Going forward, we aim to optimize the hyperparameters and improve the performance of our implementation

This approach is still valuable for understanding Q-learning adaptations to continuous control problems.

# Second Approach
We also referred to this repository: https://github.com/curiosity-creates/inverted_pendulum/blob/main/gym_inverted_pendulum.py  for solving the CartPole problem. In this case, DQN can be applied directly because the environment’s action space is already discrete — the agent only needs to decide whether to push the cart left or right.

In contrast, the Pendulum-v1 environment has a continuous action space, which is not compatible with DQN out of the box. To address this, we discretized the continuous actions into fixed torque levels, allowing DQN to be used.
