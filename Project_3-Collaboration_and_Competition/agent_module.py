from collections import deque, namedtuple

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model_module import Actor, Critic

import random
import copy

UPDATE_EVERY = 10
UPDATE_TIMES = 20
GAMMA = 0.95
TAU = 1e-3

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0

EPSILON = 1.0   # Exploration parameter - controls how much noise to add to actions when exploring
EPSILON_DECAY = 1e-6

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, seed, state_size, action_size, add_noise=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Actor - Local and Target
        self.actor_local = Actor(seed, state_size, action_size).to(device)
        self.actor_target = Actor(seed, state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)
        
        # Critic - Local and Target
        self.critic_local = Critic(seed, state_size, action_size).to(device)
        self.critic_target = Critic(seed, state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise 
        self.epsilon = EPSILON
        self.noise = OUNoise(seed, action_size)
        self.add_noise = add_noise
        
        # Replay Buffer
        self.memory = ReplayBuffer(seed, BUFFER_SIZE, BATCH_SIZE)
        
        # Steps to make learning step each UPDATE_EVERY steps 
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(UPDATE_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state):
        
        state = torch.from_numpy(state).float().to(device) 
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        self.epsilon -= EPSILON_DECAY
        if self.add_noise:
            action += np.maximum(self.epsilon, 0.2) * self.noise.sample()
        
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences
        
            # Train Critic 
        # Loss - mse of target q (reward + target_critic-predicted value for next state and next action given by target actor)
        # Next actions from target actor
        next_actions = self.actor_target(next_states)
        # Target_critic-predicted value for next state and next action
        next_values = self.critic_target(next_states, next_actions)
        # Target Values of current state
        target_values = rewards + (GAMMA * next_values * (1-dones))
        
        # Predicted values of current state
        predicted_values = self.critic_local(states, actions)
        
        # Loss function (MSE)
        loss = F.mse_loss(predicted_values, target_values)
        
        # Compute Gradient of Loss function
        self.critic_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradient 
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        
        self.critic_optimizer.step() # Make step in direction of minimum loss
        
            # Train Actor
        # Need to take actions to maximize average value (score) predicted by local critic for all actual states 
        actions_pred = self.actor_local(states)
        # HERE WE MIX IN NN OUTPUT OF LOCAL ACTOR TO NN OF LOCAL CRITIC
        actor_loss = -self.critic_local(states, actions_pred).mean() 
        
        # Compute Gradient of Loss function
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
            # Soft update Target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model, tau = TAU):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data) 

class OUNoise():
    def __init__(self, seed, action_size, mu=0., theta=0.15, sigma=0.2):
        
        self.seed = random.seed(seed)
        
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        
        self.reset()  # Reset noise after each episode 
        
    def reset(self):
        self.state = copy.copy(self.mu)
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state
    
class ReplayBuffer():
    def __init__(self, seed, buffer_size, batch_size):
        
        self.seed = random.seed(seed)
        
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
        