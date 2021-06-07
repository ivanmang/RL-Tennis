# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import Agent
from buffer import ReplayBuffer
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 1        # how often to update the network

class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(num_agents)]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=random_seed)
        self.t_step = 0
        

    def act(self, states, add_noise=True):
        """get actions of all the agents in the MADDPG object"""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i], add_noise)
            actions.append(action)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for agent in self.agents:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)

    def learn(self, experiences, agent, gamma=GAMMA):
        agent.learn(experiences, gamma)
            
    
    def reset(self):
        for agent in self.agents:
            agent.noise.reset()