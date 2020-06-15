# Report

## Learning Algorithm

Actor-Critic Deep Q-Network algorithm is implemented. Current and Target versions are implemented. Critic learns to predict 
value of current state. Actor Network chooses actions and learns to predict state value at the update step by minimizing TD Error 
(dicrepancy between Actor's prediction and sum of Immediate Reward and State Value given by Critic). 

Exploration is determined by adding Orstein-Uhlenbeck noise to continuous action. Reset parameter returns noise to 0 each 100 steps.
Number of updates per 20 learning steps is set to 10 in order to fill replay buffer with new samples before sampling repeatedly.
Remaining parameters are set following [DDPG paper](https://arxiv.org/abs/1509.02971).

## Ideas for Future Work

Apply Proximal Policy optmization, A2C algorithm with importance sampling, Priority Replay.
