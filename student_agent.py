import numpy as np
import pickle
import random
import gym
import os

def load_data():
    if os.path.exists("q_table.pkl"):
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    else: 
        q_table = {}
    return q_table

# Constants
decay_rate = 0.99999
alpha = 0.1
gamma = 0.99
action_space = [0, 1, 2, 3, 4, 5]

def get_action(obs):

    # Load pickled Q table
    q_table = load_data()

    # Positions
    taxi_pos = (obs[0], obs[1])
    station_pos = [(obs[i], obs[i+1]) for i in range(2, 10, 2)]
    obs_north_pos, obs_south_pos, obs_east_pos, obs_west_pos = obs[10:14]
    passenger_look, drop_look = obs[14], obs[15] 

    # Grid size
    m = station_pos[-1][0]

    # Storing state in function parameters between iterations, here we load them in
    epsilon = getattr(get_action, "epsilon", 0.7)
    prev_taxi_pos = getattr(get_action, "prev_taxi_pos", None)
    has_passenger = getattr(get_action, "has_passenger", False)
    visited_stations = getattr(get_action, "visited_stations", set())
    has_seen_passenger = getattr(get_action, "has_seen_passenger", False)
    has_reached_dropoff = getattr(get_action, "has_reached_dropoff", False)
    prev_action = getattr(get_action, "prev_action", None)

    # Infer passenger status
    if prev_taxi_pos in station_pos and taxi_pos != prev_taxi_pos and has_passenger is False:
        has_passenger = True

    # Store for next time
    get_action.prev_taxi_pos = taxi_pos
    get_action.has_passenger = has_passenger

    # State to index Q-table
    state = obs[0:2]  + obs[10:16] + (has_passenger,)
    if state not in q_table:
        q_table[state] = np.zeros(6)

    # Choose action
    if np.random.rand() < epsilon: 
        action = np.random.choice(action_space)
    else:
        max_q_value = np.max(q_table[state])
        best_actions = [i for i, q in enumerate(q_table[state]) if q == max_q_value]
        action = np.random.choice(best_actions)

    reward = 0
    next_pos = [taxi_pos[0], taxi_pos[1]]

    # Bounds checking
    if action in [0, 1, 2, 3]:
        if not obs_south_pos and action == 0 and next_pos[0] < m - 1:
            next_pos[0] = next_pos[0] + 1
        elif not obs_north_pos and action == 1 and next_pos[0] > 0:
            next_pos[0] = next_pos[0] - 1
        elif not obs_east_pos and action == 2 and next_pos[1] < m - 1:
            next_pos[1] = next_pos[1] + 1 
        elif not obs_west_pos and action == 3 and next_pos[1] > 0:
            next_pos[1] = next_pos[1] - 1
        else:
            reward -= 1
        
    taxi_pos = tuple(next_pos)

    get_action.prev_action = action

    # Penalize for backtracking
    if prev_taxi_pos == taxi_pos:
        reward -= 0.2
    elif prev_taxi_pos == (next_pos[0], next_pos[1]):
        reward -= 0.1
            
    # Pickup passsenger reward shaping
    if action == 4:
        if not has_passenger and taxi_pos in station_pos and passenger_look:
            has_passenger = True
            get_action.has_passenger = has_passenger
            reward += 30 
        elif taxi_pos not in station_pos:
            reward -= 1
        elif has_passenger:
            reward -= 2
        else:
            reward -= 3

    # Dropoff passenger reward shaping
    elif action == 5:
        if has_passenger and taxi_pos in station_pos and drop_look:
            reward += 60
            has_passenger = False
            get_action.has_passenger = has_passenger
        elif taxi_pos not in station_pos:
            reward -= 2
        elif not has_passenger:
            reward -= 3
        else:   
            reward -= 5
    
    # Other reward shaping
    if taxi_pos in station_pos:
        if passenger_look and not has_seen_passenger:
            reward += 5
            get_action.has_seen_passenger = True

        if drop_look and not has_reached_dropoff:
            reward += 10
            get_action.has_reached_dropoff = True
        if taxi_pos not in visited_stations:
            visited_stations.add(taxi_pos)
            reward += 5
            get_action.visited_stations = visited_stations
        else:
            reward -= 0.5
    
    next_state = obs[0:2]  + obs[10:16] + (has_passenger,)

    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(action_space), dtype=np.float32)

    q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

    epsilon = max(epsilon * decay_rate, 0.1)

    get_action.epsilon = epsilon

    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)

    return action

