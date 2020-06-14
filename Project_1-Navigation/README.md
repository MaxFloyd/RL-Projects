# RL Project 1 - Navigation

## Project Details 

Agent navigates in a square world where yellow and blue bananas spawn occasionally. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas.

![Environment](https://github.com/MaxFloyd/RL-Project-1---Navigation/blob/master/banana.gif)

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Dependencies

Clone [DLRND repository](https://github.com/udacity/deep-reinforcement-learning) and download dependencies as outlined [here](https://github.com/udacity/deep-reinforcement-learning#dependencies)  

### Replicating (Instructions) 

Copy files from [this repository](https://github.com/MaxFloyd/RL-Project-1---Navigation) into p1_navigation directory in cloned DLRND repository. 

For trained agent run all cells in run.ipynb to see performance of untrained agent first and then performance of the trained version.

To train on own machine run train.ipynb.

### Repository contents 

agent_module.py - agent class used for training and testing.  
model_module.py - classes for neural nets architecture.  
run.ipynb - notebook for testing trained agent's performance.  
train.ipynb - notebook for training the agent.  


