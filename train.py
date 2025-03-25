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
        
        # Q-table stored as a dictionary: obs -> np.array of shape (6,) for Q-values
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
            # Collect all actions whose Q-value == max_q_value so we can pick randomly among ties
            best_actions = [a for a, q in enumerate(q_values) if q == max_q_value]
            return random.choice(best_actions)

    def update(self, obs, action, reward, next_obs, done):
        """
        Update the Q-table using the standard Q-learning update rule.
        """
        current_q = self.get_q_values(obs)
        next_q = self.get_q_values(next_obs)
        
        if done:
            td_target = reward
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
    decay=0.9999
):
    """
    Train a Q-learning taxi agent in SimpleTaxiEnv with reward shaping 
    (inspired by your attached snippet).
    """
    agent = QLearningAgent(alpha, gamma, epsilon, epsilon_min, decay)
    all_rewards = []

    for ep in range(episodes):
        # Random grid size from 5 to 10 (as you do)
        grid_size = random.randint(5, 10)
        env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=200)

        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        # =========================
        # Variables for shaping
        # =========================
        passenger_in_taxi = False
        visited_stations = set() 
        visited_passenger_station = False
        visited_dropoff_station = False
        previous_taxi_pos = (obs[0], obs[1])  # (row, col)
        previous_action = None

        for step in range(max_steps):
            # 1) Choose action from Q-table (epsilon-greedy)
            action = agent.choose_action(obs)

            # 2) Environment step
            next_obs, base_reward, done, _ = env.step(action)
            
            # 3) Add the snippet’s shaping
            shaping_reward = 0.0

            # Current taxi pos & next taxi pos
            taxi_row, taxi_col = obs[0], obs[1]
            next_taxi_row, next_taxi_col = next_obs[0], next_obs[1]

            # Stations:
            station_positions = [
                (obs[2],  obs[3]),
                (obs[4],  obs[5]),
                (obs[6],  obs[7]),
                (obs[8],  obs[9])
            ]
            # Booleans for obstacles, passenger_look, drop_look
            obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10], obs[11], obs[12], obs[13]
            passenger_look, drop_look = obs[14], obs[15]

            # --------------------------------------------------
            #  (A) Movement shaping
            # --------------------------------------------------
            # If we do a movement action
            if action in [0, 1, 2, 3]:
                # If the next position == old position, then agent tried to move but failed
                # or moved out of bounds/ into obstacle
                if (next_taxi_row, next_taxi_col) == (taxi_row, taxi_col):
                    # We can penalize it for "wasted move"
                    shaping_reward -= 1.0
                else:
                    # Prevent flip flopping back and forth 
                    if previous_action == 0 and action == 1:
                        shaping_reward -= 0.5
                    elif previous_action == 1 and action == 0:
                        shaping_reward -= 0.5
                    elif previous_action == 2 and action == 3:
                        shaping_reward -= 0.5
                    elif previous_action == 2 and action == 3:
                        shaping_reward -= 0.5
                    else:
                        shaping_reward -= 0.05

                    # If the agent “backtracked” immediately:
                    if (next_taxi_row, next_taxi_col) == previous_taxi_pos:
                        shaping_reward -= 0.4

            # --------------------------------------------------
            #  (B) Pickup action (4)
            # --------------------------------------------------
            if action == 4:
                # If not carrying passenger yet, and the agent is actually at a station w/ passenger_look
                # => success
                if (not passenger_in_taxi) and (taxi_row, taxi_col) in station_positions and passenger_look:
                    shaping_reward += 30
                    passenger_in_taxi = True
                else:
                    # check the snippet logic
                    if (taxi_row, taxi_col) not in station_positions:
                        shaping_reward -= 1
                    elif passenger_in_taxi:
                        shaping_reward -= 2
                    else:
                        shaping_reward -= 3

            # --------------------------------------------------
            #  (C) Dropoff action (5)
            # --------------------------------------------------
            if action == 5:
                # If we are carrying passenger, next station has drop_look => success
                if passenger_in_taxi and (taxi_row, taxi_col) in station_positions and drop_look:
                    shaping_reward += 100
                    passenger_in_taxi = False
                else:
                    if (taxi_row, taxi_col) not in station_positions:
                        shaping_reward -= 2
                    elif not passenger_in_taxi:
                        shaping_reward -= 3
                    else:
                        shaping_reward -= 5

            # --------------------------------------------------
            #  (D) Visiting station shaping
            # --------------------------------------------------
            if (taxi_row, taxi_col) in station_positions:
                # If passenger_look & not visited_passenger_station => +5
                if passenger_look and not visited_passenger_station:
                    shaping_reward += 5
                    visited_passenger_station = True
                # If drop_look & not visited_dropoff_station => +10
                if drop_look and not visited_dropoff_station:
                    shaping_reward += 10
                    visited_dropoff_station = True

                # If this station is newly visited => +5
                if (taxi_row, taxi_col) not in visited_stations:
                    visited_stations.add((taxi_row, taxi_col))
                    shaping_reward += 5
                else:
                    # Station re-visited => -0.5
                    shaping_reward -= 0.5

            # Summation of base env reward + shaping
            total_step_reward = base_reward + shaping_reward

            # 4) Q-learning update
            agent.update(obs, action, total_step_reward, next_obs, done)

            # 5) Bookkeeping
            total_reward += total_step_reward
            obs = next_obs
            previous_taxi_pos = (next_taxi_row, next_taxi_col)
            previous_action = action

            if done:
                break

        agent.decay_epsilon()
        all_rewards.append(base_reward)

        if (ep + 1) % 500 == 0:
            avg_reward = np.mean(all_rewards[-500:])
            print(f"Episode {ep+1}/{episodes} - Avg Reward (last 500 eps): {avg_reward:.2f}"
                  f" - Epsilon: {agent.epsilon:.3f}")

    # Save final Q-table
    agent.save('q_table.pkl')
    print("Training complete. Q-table saved as q_table.pkl.")
    return agent


if __name__ == "__main__":
    # Example usage: adjust episodes and other hyperparams as needed
    trained_agent = train_agent(
        episodes=150000,
        max_steps=200,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        decay=0.99999
    )
