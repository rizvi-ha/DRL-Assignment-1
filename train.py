import numpy as np
import pickle
import random
import time
from simple_custom_taxi_env import SimpleTaxiEnv

class QLearningAgent:
    """
    A simple Q-learning agent that stores its Q-table in a Python dictionary.
    The key for each entry in the Q-table is the environment's 'obs' tuple,
    and the value is a length-6 array corresponding to the 6 possible actions.
    """
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, decay=0.9995):
        """
        alpha: learning rate
        gamma: discount factor
        epsilon: initial epsilon for epsilon-greedy
        epsilon_min: minimum possible epsilon value
        decay: decay rate applied to epsilon after each episode
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        
        # Q-table stored as a dictionary of obs -> Q-value array of shape (6,)
        self.q_table = {}

    def get_q_values(self, obs):
        """
        Return the Q-values for the given obs.
        If obs not found, initialize with zeros.
        """
        if obs not in self.q_table:
            self.q_table[obs] = np.zeros(6, dtype=np.float32)
        return self.q_table[obs]

    def choose_action(self, obs):
        """
        Choose an action using epsilon-greedy. 
        With probability epsilon, pick a random action (0~5).
        Otherwise, pick the best action from the Q-table.
        """
        if random.random() < self.epsilon:
            return random.randint(0, 5)
        else:
            q_values = self.get_q_values(obs)
            max_q_value = np.max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q_value]
            return random.choice(best_actions)

    def update(self, obs, action, reward, next_obs, done):
        """
        Update the Q-table using the standard Q-learning update rule.
        """
        current_q = self.get_q_values(obs)
        next_q = self.get_q_values(next_obs)
        
        if done:
            td_target = reward  # no future reward if episode ends
        else:
            td_target = reward + self.gamma * np.max(next_q)
        
        # Q-learning update
        current_q[action] += self.alpha * (td_target - current_q[action])

    def decay_epsilon(self):
        """
        Decay epsilon after each episode, ensuring it does not go below epsilon_min.
        """
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)

    def save(self, filename='q_table.pkl'):
        """
        Save the Q-table to file using pickle.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

def train_agent(
    episodes=5000,
    max_steps=500,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.05,
    decay=0.9995
):
    """
    Train a Q-learning taxi agent in SimpleTaxiEnv.
    The environment is random each episode to encourage generalization.
    """
    # You may alter the grid_size and fuel_limit to approximate the real environment more closely.
    # lets have grid size be random between 5 and 10
    grid_size = random.randint(5, 10)
    env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=2000)  
    agent = QLearningAgent(alpha, gamma, epsilon, epsilon_min, decay)

    all_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        for step in range(max_steps):
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

            if done:
                break
        
        # Decay epsilon after each episode
        agent.decay_epsilon()
        all_rewards.append(total_reward)

        # Print progress every 500 episodes (optional)
        if (ep + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-500:])
            print(f"Episode {ep+1}/{episodes} - Avg Reward (last 500 eps): {avg_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    # Save the trained Q-table
    agent.save('q_table.pkl')
    print("Training complete. Q-table saved as q_table.pkl.")
    return agent

if __name__ == "__main__":
    # Example usage: adjust episodes and other hyperparams as needed
    trained_agent = train_agent(
        episodes=10000,
        max_steps=2000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        decay=0.9995
    )
