# GridWorld-QLearning
Concurrent Q-Learning agent implementation for a gridworld environment with obstacles and stochastic mechanics. Built for Programming Technologies course.
This project implements a tabular Reinforcement Learning agent trained to navigate a complex gridworld environment. Developed for the Programming Technologies course at Azerbaijan State Oil and Industry University (ASOIU).
🎯 OverviewThe objective is to train an agent to find the optimal path from a starting point (0,0) to a goal (4,4) while avoiding static obstacles (walls) and penalty zones (traps). The system utilizes Concurrent Programming to accelerate the learning process through parallel experience collection.
🛠️ Key Technical FeaturesQ-Learning Algorithm: Implementation of the off-policy RL algorithm using the Bellman Equation.
Concurrency: Multi-threaded training runner that allows multiple environment workers to update a global Q-table simultaneously.
Thread Safety: Utilization of threading.Lock to ensure safe aggregation of Q-table updates and prevent race conditions.
Stochastic Mechanics: A "Slippery Tile" feature that adds a probability of random movement, testing the agent's robustness.
Exploration Strategy: $\epsilon$-greedy strategy with an exponential decay schedule to balance discovery and exploitation.
Data Logging: Automatic export of training metrics (Episode vs. Reward) to CSV format for performance visualization.
🏗️ Environment SpecificationsGrid Size: 5 X 5 
Start State: (0,0)
Goal State: (4,4) [Reward: +100]
Traps: (1,1), (3,2) [Penalty: -50]
Walls: (2,2), (2,3) [Inaccessible]
Stochasticity: 10\% slip probability.
🚀 Getting StartedPrerequisitesPython 3.8+
NumPy
Installation
1.Clone the repository:git clone https://github.com/YOUR_USERNAME/GridWorld-QLearning.git
2.Install dependencies:pip install numpy
Running the Simulator:
To train the agent and see the live demo, execute:python main.py
📈 Results & Validation:
The agent's progress is logged in training_stats.csv. During the live demo, the program prints the Q-values for each state, providing transparency into the agent's decision-making process.
