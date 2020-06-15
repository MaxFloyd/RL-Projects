# Report

## Learning Algorithm

Actor-Critic Deep Q-Network algorithm is implemented. Current and Target versions are implemented. Critic learns to predict 
value of current state. Actor Network chooses actions and learns to take best action at the update step by minimizing TD Error 
(dicrepancy between Actor's prediction and sum of Immediate Reward and State Value given by Critic). 

Exploration is determined by adding Orstein-Uhlenbeck noise to continuous action. Reset parameter returns noise to 0 each 100 steps.
Number of updates per 20 learning steps is set to 10 in order to fill replay buffer with new samples before sampling repeatedly.
Remaining parameters are set following [DDPG paper](https://arxiv.org/abs/1509.02971).

## Net Architecture

3-layer: 256, 128, 128 units following DDPG paper architecture.


## Ideas for Future Work

Apply Proximal Policy optmization, A2C algorithm with importance sampling, Priority Replay.

## Rewards Plot

![Rewards Plot](https://github.com/MaxFloyd/RL-Projects/blob/master/Project_2-Continuous_Control/reward_plot.png)
