import numpy as np
import pickle
import os

def load_q_table():
    if os.path.exists("q_table.pkl"):
        with open("q_table.pkl", "rb") as f:
            q_table = pickle.load(f)
    else:
        # Fallback: if no file found, use an empty dict (or you could use a random approach).
        q_table = {}

    return q_table

def get_q_values(obs, q_table):
    """
    For states not in our Q-table, we initialize an array of zeros for the 6 actions.
    This is crucial when facing new states in dynamic environments.
    """
    if obs not in q_table:
        q_table[obs] = np.zeros(6, dtype=np.float32)
    return q_table[obs]

def get_action(obs, debug=False):
    """
    The environment calls this function every time it needs an action from the agent.
    
    obs: a tuple containing the environment's current observation.
         e.g. (taxi_row, taxi_col, stationR_row, stationR_col, ..., passenger_look, destination_look)
         
    Return an integer 0~5 as your chosen action:
      0 = Move South
      1 = Move North
      2 = Move East
      3 = Move West
      4 = Pick Up
      5 = Drop Off
    """
    # Epsilon is not used in evaluation—always pick the best known action from Q-table.
    q_table = load_q_table()
    q_values = get_q_values(obs, q_table)
    max_q_value = np.max(q_values)
    best_actions = [a for a, q in enumerate(q_values) if q == max_q_value]
    best_action = np.random.choice(best_actions)
    return best_action
