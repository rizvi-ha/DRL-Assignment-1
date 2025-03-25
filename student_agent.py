import numpy as np
import pickle
import os

# Global variable to track whether passenger is currently on the taxi.
passenger_in_taxi = False

# We'll load the Q-table once and store it here (to avoid re-loading from disk on every call).
q_table = None

def load_q_table():
    """
    Load Q-table from file if present, otherwise use an empty dictionary.
    We'll store it in the global 'q_table' variable so it's only loaded once.
    """
    global q_table
    if q_table is None:
        if os.path.exists("q_table.pkl"):
            with open("q_table.pkl", "rb") as f:
                q_table = pickle.load(f)
        else:
            q_table = {}
    return q_table

def extract_state(obs, passenger_on):
    """
    Replicates the same state extraction as in train.py:
    (taxi_row, taxi_col, pass_on_int, wall_down, wall_up, wall_right, wall_left, passenger_look, destination_look)
    """
    taxi_row, taxi_col = obs[0], obs[1]

    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east  = obs[12]
    obstacle_west  = obs[13]

    passenger_look = obs[14]
    destination_look = obs[15]

    pass_on_int = 1 if passenger_on else 0

    wall_down = 1 if obstacle_south else 0
    wall_up   = 1 if obstacle_north else 0
    wall_right= 1 if obstacle_east else 0
    wall_left = 1 if obstacle_west else 0

    return (
        taxi_row, taxi_col, 
        pass_on_int, 
        wall_down, wall_up, wall_right, wall_left, 
        passenger_look, destination_look
    )

def get_q_values(state):
    """
    Retrieve Q-values for a given state from the Q-table.
    If not present, initialize them with [0, 0, 0, 0, -1, -1].
    """
    global q_table
    if state not in q_table:
        q_table[state] = np.array([0.5, 0.5, 0.5, 0.5, -6, -6], dtype=np.float32)
    return q_table[state]

def get_action(obs, debug=False):
    """
    The environment calls this function every time it needs an action from the agent.
    We:
      1) Convert (obs, passenger_in_taxi) to a 'state' using extract_state.
      2) Lookup the Q-values for that state.
      3) Choose the action with the highest Q-value (random tie-break).
      4) Naively update our passenger_in_taxi flag if action==Pickup or action==Dropoff.
    """
    global passenger_in_taxi

    # 1) Ensure our Q-table is loaded
    load_q_table()

    # 2) Extract the 9-tuple state from obs & passenger_in_taxi
    state = extract_state(obs, passenger_in_taxi)
    
    # 3) Retrieve Q-values and select the best action (random tie-break)
    q_values = get_q_values(state)
    max_q = np.max(q_values)
    best_actions = [a for a, val in enumerate(q_values) if val == max_q]
    action = np.random.choice(best_actions)

    # 4) Simple/naive passenger_in_taxi update 
    #    (matching the condition for "successful" pickup or dropoff from the training logic)
    passenger_look = obs[14]
    destination_look = obs[15]

    # Pickup
    if action == 4:
        if (not passenger_in_taxi) and passenger_look:
            passenger_in_taxi = True
    # Dropoff
    elif action == 5:
        if passenger_in_taxi and destination_look:
            passenger_in_taxi = False

    return action
