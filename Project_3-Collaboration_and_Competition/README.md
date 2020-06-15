# Project 3 - Collaboration and Competition

## Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![Tennis Environment](https://github.com/MaxFloyd/RL-Projects/blob/master/Project_3-Collaboration_and_Competition/tennis_pic.png)

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Dependencies

Clone [DLRND repository](https://github.com/udacity/deep-reinforcement-learning) and download dependencies as outlined [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)  

### Replicating (Instructions) 

Copy files from [this repository](https://github.com/MaxFloyd/RL-Projects/tree/master/Project_3-Collaboration_and_Competition)  into p2_continuous-control directory in cloned DLRND repository. 

Download Unity Environment ([MacOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip), [Win](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)). Place unzipped file into p2_continuous-control directory in cloned DLRND repository.

For trained agent run all cells in run.ipynb to see performance of untrained agent first and then performance of the trained version.

To train on own machine run train.ipynb.
