# Report

## Learning Algorithm

Actor-Critic Deep Q-Network algorithm is implemented. One pair of Actor-Critic Networks is implemented for each Agent.
Current and Target versions are implemented both for Actor and Critic for each Agent. 
Critic learns to predict value of current state. 
Actor Network chooses actions and learns to take best action at the update step by minimizing TD Error (dicrepancy between Actor's prediction and sum of Immediate Reward and State Value given by Critic).

Parameters are set following [DDPG paper](https://arxiv.org/abs/1509.02971).

## Net Architecture

3-layer: 256, 128, 128 units following DDPG paper architecture.

## Ideas for Future Work

Apply Proximal Policy optmization, A2C algorithm with importance sampling, Priority Replay.
