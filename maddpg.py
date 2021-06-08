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
TAU = 1e-3  

class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        self.agents = [Agent(i, state_size, action_size, random_seed) for i in range(num_agents)]
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
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for agent in self.agents:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)
        for agent in self.agents:
            agent.soft_update(agent.critic_local,agent.critic_target,TAU)
            agent.soft_update(agent.actor_local,agent.actor_target,TAU)    

    def learn(self, experiences, agent, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences
        
        #Feed the states of that agent into the actor of that agent
        actions_next = [a.actor_target(next_states.index_select(1, torch.tensor([i])).squeeze(1)) for i, a in enumerate(self.agents)]
        
        actions_pred = [a.actor_local(states.index_select(1, torch.tensor([i])).squeeze(1)) for i, a in enumerate(self.agents)]
        # Combine the actions from both agents in one tensor
        actions_next = torch.cat(actions_next, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        agent.learn(experiences, actions_next, actions_pred, gamma)
            
    
    def reset(self):
        for agent in self.agents:
            agent.noise.reset()