# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import Agent
from buffer import ReplayBuffer
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 2        # how often to update the network


class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        super(MADDPG, self).__init__()
        self.agents = [Agent(i, state_size, action_size, random_seed) for i in range(num_agents)]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed=random_seed)
        self.t_step = 0
        
    
    def act(self, states, add_noise=True):
        """get actions of all the agents in the MADDPG object"""
        actions = []
        for agent, state in zip(self.agents, states):
            actions.append(agent.act(np.expand_dims(state, axis=0), add_noise).squeeze(0))
        return np.stack(actions)

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

    def learn(self, experiences, agent, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences
        
        #Feed the states of that agent into the actor of that agent
        actions_next = [a.actor_target(next_states.index_select(1, torch.tensor([i])).squeeze(1)) for i, a in enumerate(self.agents)]
        
        # Combine the actions from both agents in one tensor
        actions_next = torch.cat(actions_next, dim=1).to(device)


        # Get the predicted actions from the actor
        agent_action_pred = agent.actor_local(states.index_select(1, agent.id).squeeze(1))
        
        actions_pred = [agent_action_pred if i==agent.id.numpy()[0] else actions.index_select(1, torch.tensor([i]).to(device)).squeeze(1) for i, _ in enumerate(self.agents)]

        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        agent.learn(experiences, actions_next, actions_pred, gamma)
            
    
    def reset(self):
        for agent in self.agents:
            agent.noise.reset()
            
    def save_weight(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{i}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_{i}.pth') 