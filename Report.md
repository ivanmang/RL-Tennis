# Project 3 Report

## Introduction
In this project, an agent is trained to move a double-joined arm to target locations under the  [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.
It is trained using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm. The environment contains 2 identical agents, each have its own independent observations and actions.

## Learning Algorithm
Multi-Agent Deep Deterministic Policy Gradient ([MADDPG](https://arxiv.org/pdf/1706.02275.pdf)) algorithm is a algorithm which contains mutiple [DDPG](https://arxiv.org/abs/1509.02971) agents. Just like a normal DDPG agent, The Actor Network output the best possible actions from the current state of the environment. The Critic Network predict the Q-values from the states and actions. In [MADDPG](https://arxiv.org/pdf/1706.02275.pdf), since there are states from mutiple-agents, the actor network will only take the observations (state and action) from that specific agents; while critic network will take observations from all agents. This allow the critics of each agent aware of other agents behaviour and therefore make a better prediction.

### `maddpg.py`
This file contains multiple DDPG agents and a shared replay buffer. This provide a platform to handle observations involve multiple agents. The both predicted actions of each agents is combined here and passed to each DDPG's agent critic network. 

#### Hyperparameter
`BUFFER_SIZE = int(1e6)  # replay buffer size`

`BATCH_SIZE = 256        # minibatch size`

`GAMMA = 0.99            # discount factor`

`UPDATE_EVERY = 2        # how often to update the network`

`LR_ACTOR = 1e-4         # learning rate of the actor `

`LR_CRITIC = 5e-4        # learning rate of the critic`

`WEIGHT_DECAY = 0        # L2 weight decay`

` TAU = 0.9               # for soft update of target parameters`

### `ddpg_agent.py`
This file defines the agent using [DDPG](https://arxiv.org/abs/1509.02971). It compose of two main network Actor Network (w/ Target Network) and Critic Network (w/ Target Network). Since mutiple-agents is involved, the critic network have a input of the size of both states, actions from both agents, while actor network have the input from its own agent's states only. Ornstein-Uhlenbeck noise is added to the actions to increase the "exploration" part the agent, with a parameter where mu=0., theta=0.15, sigma=0.2.


### `model.py`
This file cotain the architecture of the network of the Actor and Critic model. 

#### `Actor`
The input of the network is 24 as the state size is 24. The first hidden layer has 256 nodes, followed by a Batch Normalization layer. Then, the second hidden layer contains 256 nodes. The output layer is 2 which is the action size. All these layers are separated by Rectifier Linear Units (ReLu) except the output layer using the `tanh` function. 

#### `Critic`
The input of the network is 48 as the 2 agent's state size 2x24. The first hidden layer has 256 nodes, followed by a Batch Normalization layer.Then, we concatenate this layer's output with the actions which is the actions from both agents 2x2 = 4. Then, the second hidden layer contains 256 nodes. The output layer have 1 node which is the Q-values. All these layers are separated by Rectifier Linear Units (ReLu). 

#### Other training hyperparameter
`n_episodes=5000 #Number of episodes we trained`


### Result
