import numpy as np
import random
import threading
import time
import json
import csv
from copy import deepcopy

# --- CONFIGURATION ---
GRID_SIZE = 5
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = ["↑", "↓", "←", "→"]

class GridWorld:
    def __init__(self, size=5, stochastic_prob=0.1):
        self.size = size
        self.stochastic_prob = stochastic_prob
        self.goal = (4, 4)
        self.traps = [(1, 1), (3, 2)]
        self.walls = [(2, 2), (2, 3)]
        self.start_pos = (0, 0)
        self.state = self.start_pos

    def reset(self):
        self.state = self.start_pos
        return self.state

    def step(self, action):
        # Stochastic mechanic: "Slippery Tile"
        if random.random() < self.stochastic_prob:
            action = random.choice(ACTIONS)

        r, c = self.state
        if action == UP: r = max(0, r - 1)
        elif action == DOWN: r = min(self.size - 1, r + 1)
        elif action == LEFT: c = max(0, c - 1)
        elif action == RIGHT: c = min(self.size - 1, c + 1)

        # Check for walls
        if (r, c) in self.walls:
            new_state = self.state
        else:
            new_state = (r, c)

        self.state = new_state

        # Rewards and Terminal State
        if self.state == self.goal:
            return self.state, 100, True
        if self.state in self.traps:
            return self.state, -50, True
        
        return self.state, -1, False # Step penalty to encourage speed

class QLearningAgent:
    def __init__(self, size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = np.zeros((size, size, len(ACTIONS)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lock = threading.Lock() # For safe concurrent updates
        self.history = []

    def get_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        with self.lock: # Ensure thread-safety
            old_value = self.q_table[state][action]
            next_max = np.max(self.q_table[next_state])
            
            # Q-Learning Update Rule
            new_value = old_value + self.alpha * (reward + self.gamma * next_max * (1 - done) - old_value)
            self.q_table[state][action] = new_value

def worker_thread(agent, num_episodes, thread_id, results):
    env = GridWorld(size=GRID_SIZE)
    episode_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        agent.epsilon = max(0.01, agent.epsilon * agent.epsilon_decay)
    
    results[thread_id] = episode_rewards

def train_concurrent(num_workers=4, ep_per_worker=250):
    agent = QLearningAgent(GRID_SIZE)
    threads = []
    results = [None] * num_workers

    print(f"🚀 Starting concurrent training on {num_workers} workers...")
    
    for i in range(num_workers):
        t = threading.Thread(target=worker_thread, args=(agent, ep_per_worker, i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Combine metrics
    all_rewards = [reward for worker_res in results for reward in worker_res]
    
    # Export metrics
    with open('training_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])
        for i, r in enumerate(all_rewards):
            writer.writerow([i, r])
    
    print("✅ Training complete. Stats saved to training_stats.csv")
    return agent

def run_demo(agent):
    print("\n--- 🏁 LIVE AGENT DEMO ---")
    env = GridWorld(size=GRID_SIZE, stochastic_prob=0) # Deterministic for demo
    state = env.reset()
    done = False
    
    path_grid = np.full((GRID_SIZE, GRID_SIZE), ".")
    for w in env.walls: path_grid[w] = "W"
    for t in env.traps: path_grid[t] = "T"
    path_grid[env.goal] = "G"

    while not done:
        r, c = state
        q_values = agent.q_table[state]
        action = agent.get_action(state, explore=False)
        
        print(f"\nAgent at {state}")
        print(f"Q-values: UP:{q_values[0]:.1f} DOWN:{q_values[1]:.1f} LEFT:{q_values[2]:.1f} RIGHT:{q_values[3]:.1f}")
        print(f"Action chosen: {ACTION_NAMES[action]} (Max Q-value)")
        
        state, _, done = env.step(action)
        if state == env.goal:
            print("Target reached! 🏆")
        elif state in env.traps:
            print("Agent fell in a trap! 💀")

if __name__ == "__main__":
    trained_agent = train_concurrent()
    run_demo(trained_agent)