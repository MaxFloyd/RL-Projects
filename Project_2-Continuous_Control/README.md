# RL Project 2 - Continuous Control

## Project Details

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

![Reacher Environment](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average of 20 actors reward is at least +30 over 100 episodes.


### Dependencies

Clone [DLRND repository](https://github.com/udacity/deep-reinforcement-learning) and download dependencies as outlined [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)  

### Replicating (Instructions) 

Copy files from [this repository](https://github.com/MaxFloyd/RL-Projects/tree/master/Project_2-Continuous_Control) into p2_continuous-control directory in cloned DLRND repository. 

For trained agent run all cells in run.ipynb to see performance of untrained agent first and then performance of the trained version.

To train on own machine run train.ipynb.
