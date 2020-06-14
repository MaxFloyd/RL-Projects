import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque, namedtuple
import random

from model_module import DQ_Network

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self, state_size, action_size, eps, eps_decay, buff_size, samp_size, learn_freq, gamma, optim_lr, tau, net_params):
        """
        net_params - list of parameters to be passed to DQ_Network
        """
        # General parameters
        self.state_size = state_size
        self.action_size = action_size
        
        self.target_net = DQ_Network(state_size, action_size, *net_params).to(device)
        self.current_net = DQ_Network(state_size, action_size, *net_params).to(device)
        
        # Acion step parameters
        self.eps = eps
        self.eps_decay = eps_decay
        
        # Learning step parameters
        self.t_step = 0
        self.learn_every = learn_freq
        self.samp_size = samp_size
        self.gamma = gamma
        self.tau = tau
        
        self.optimizer = optim.Adam(params=self.current_net.parameters(), lr = optim_lr)
        
        self.buff = ReplayBuffer(buff_size, samp_size)
        
        
    def action(self, state):
        """
        Implementation of eps-greedy policy 
        
        Returns: action to take
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.current_net.eval()
        with torch.no_grad():
            actions = self.current_net(state)
        self.current_net.train()
        
        self.eps = self.eps * self.eps_decay
        if np.random.rand() > self.eps: 
            return np.argmax(actions.cpu().data.numpy())  # with prob 1-eps take max q-value action given by the current model 
        else:
            action = np.random.choice(range(self.action_size))
        return action
            
    def step(self, state, action, reward, state_next, done):
        """
        Save experience tuple, initiate learning at constant intervals
        """
        self.t_step += 1
        
        self.buff.save(state, action, reward, state_next, done)
        
        if (self.t_step + 1) % self.learn_every == 0:
            if len(self.buff.deq) > self.samp_size:
                sample = self.buff.sample()
                self.learn(sample, self.gamma) # pass vectors of states, actions, rewards and next states
        
    def learn(self, experiences, gamma):
        """
        Update weights in Neural Network
        Once updated weights in current NN, update target N
        """
        
        states, actions, rewards, states_next, dones = experiences
        
        # First get current approximation of q-values corresponding to actual actions taken during interaction
        q_current = self.current_net(states).gather(1, actions)
        
        # Get estimate of action-value function
        # Get maximum q-values in next state according to target net
        # Don't forget to detach target tensor
        q_target_next_state = self.target_net(states_next).detach().max(1)[0].unsqueeze(1)
        
        q_target = rewards + gamma*q_target_next_state*(1-dones)
        
        self.optimizer.zero_grad()
        
        loss = F.mse_loss(q_current, q_target)
        loss.backward()
        self.optimizer.step()
        
        # Update target net weights
        self.soft_update(self.target_net, self.current_net, self.tau)
        
    def soft_update(self, target_net, current_net, tau):
        """
        Update target net weights towards q_current weights according to parameter tau
        Higher tau - faster weights converge to target weights
        """
        
        for target_param, current_param in zip(target_net.parameters(), current_net.parameters()):
            target_param.data.copy_(tau*current_param.data + (1-tau)*target_param.data)
        

class ReplayBuffer():
    """
    Queue-like array which holds experience tuples and provides sampling functionality for learn step
    """
    def __init__(self, maxlen, samp_size):
        
        self.deq = deque(maxlen=maxlen)
        self.samp_size = samp_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def save(self, state, action, reward, state_next, done):
        e = self.experience(state, action, reward, state_next, done)
        self.deq.append(e)
        
    def sample(self):
        experiences = random.sample(self.deq, k=self.samp_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
        
        
        