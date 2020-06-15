# Report

### Learning Algorithm 

Deep Q Network with Replay Buffer is implemented. Agent learns optimal action-value function via Gradient Descent on Fixed Q-Targets update rule - using current and target versions of action-value-approximating Neural Net. 

Standard learning parameters implemented - 0.1 exploration with 0.999 decay and allows agent to make a substantial number of 
exploratory moves before starting to rely on learned action-value pairs entirely. Buffer size and sample size of 10000 and 64 respectively are chosen to balance representativeness and processing speed. Same is true for learning frequency of 4. Learning rate of 0.0005 is chosen by trial-and-error. Among considered rates it gives the fastest convergence to target. 

### Net Architecture

Both Actor and Critic are implemented as 3-layer (Linear) Neural Nets with 128, 256, 128 units in each layer. 
Powers of 2 are common choice for number of units. Depth of 3 is chosen heuristically yet such arrangement manages to solve
the problem in reasonable time.

### Rewards Plot

![Rewards Plot](https://github.com/MaxFloyd/RL-Projects/blob/master/Project_1-Navigation/reward_plot.png)

### Ideas for Future Work

Future directions include implementing Priority Replay and Dueling DQN. 
