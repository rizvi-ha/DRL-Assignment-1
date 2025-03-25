# simple_custom_taxi_env.py

import importlib.util
import numpy as np
import random
import time

class SimpleTaxiEnv:
    """
    A simplified Taxi environment for testing Q-learning agents offline.

    Observations (returned by get_state()):
      obs = (
         taxi_row,          # agent's row
         taxi_col,          # agent's column
         station0_row,
         station0_col,
         station1_row,
         station1_col,
         station2_row,
         station2_col,
         station3_row,
         station3_col,
         obstacle_north,    # 1 if there's a wall/obstacle or boundary above
         obstacle_south,    # 1 if there's a wall/obstacle or boundary below
         obstacle_east,     # 1 if there's a wall/obstacle or boundary to the right
         obstacle_west,     # 1 if there's a wall/obstacle or boundary to the left
         passenger_look,    # 1 if passenger is in or adjacent to the taxi position
         destination_look   # 1 if destination is in or adjacent to the taxi position
      )

    Actions:
      0 = Move South
      1 = Move North
      2 = Move East
      3 = Move West
      4 = Pickup
      5 = Dropoff
    """

    def __init__(self, grid_size=5, fuel_limit=50):
        """
        :param grid_size: Size of the square grid (e.g., 5x5).
        :param fuel_limit: Max steps before environment ends automatically.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit

        self.stations = [
            (0, 0),                      # R
            (0, self.grid_size - 1),     # G
            (self.grid_size - 1, 0),     # Y
            (self.grid_size - 1, self.grid_size - 1)  # B
        ]
        self.obstacles = set()  # Could be randomized if desired
        self.reset()

    def reset(self):
        """
        Resets the environment to a new random state.
        Returns the initial observation tuple and an empty info dict.
        """
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False

        # You can add random obstacles if desired:
        #   self.obstacles = {(1, 1), (2, 2), ...}

        # Randomly place the taxi somewhere not in obstacles or stations
        valid_positions = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.obstacles and (r, c) not in self.stations
        ]
        self.taxi_pos = random.choice(valid_positions)

        # Randomly select which station is passenger start, and which is destination
        self.passenger_loc = random.choice(self.stations)
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)

        return self.get_state(), {}

    def step(self, action):
        """
        Updates the environment state based on the chosen action (0~5).
        Returns (obs, reward, done, info).
        """
        done = False
        reward = 0.0

        # Current row/col
        row, col = self.taxi_pos

        # Movement actions
        if action == 0:  # South
            next_row = row + 1
            next_col = col
        elif action == 1:  # North
            next_row = row - 1
            next_col = col
        elif action == 2:  # East
            next_row = row
            next_col = col + 1
        elif action == 3:  # West
            next_row = row
            next_col = col - 1
        else:
            # If it's not a movement action, the taxi doesn't change position by default
            next_row, next_col = row, col

        # Check if movement is valid (not out of bounds or into an obstacle)
        if action in [0, 1, 2, 3]:
            if not self._valid_position(next_row, next_col):
                # Invalid move => penalty
                reward -= 5
                next_row, next_col = row, col  # remain in place
            else:
                # Valid move
                self.taxi_pos = (next_row, next_col)
                # If passenger is currently in the taxi, they move with the taxi
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos

        # Pickup action
        elif action == 4:
            if self.taxi_pos == self.passenger_loc and not self.passenger_picked_up:
                # Correct pickup
                self.passenger_picked_up = True
            else:
                # Wrong pickup
                reward -= 10

        # Dropoff action
        elif action == 5:
            if self.passenger_picked_up:
                # If at the correct destination, big reward => done
                if self.taxi_pos == self.destination:
                    reward += 50
                    done = True
                else:
                    # Dropped off in the wrong location
                    reward -= 10
                self.passenger_picked_up = False
            else:
                # Trying to drop off with no passenger
                reward -= 10

        # Small step penalty
        reward -= 0.1
        # Reduce fuel
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            # If out of fuel, environment terminates
            reward -= 10
            done = True

        return self.get_state(), reward, done, {}

    def get_state(self):
        """
        Returns a tuple representing the environment observation:
        (taxi_row, taxi_col,
         station0_row, station0_col,
         station1_row, station1_col,
         station2_row, station2_col,
         station3_row, station3_col,
         obstacle_north, obstacle_south, obstacle_east, obstacle_west,
         passenger_look, destination_look)
        """
        row, col = self.taxi_pos
        stations_list = list(self.stations)

        # For convenience
        # Obstacles = boundary or in self.obstacles
        obstacle_north = int(row == 0 or (row - 1, col) in self.obstacles)
        obstacle_south = int(row == self.grid_size - 1 or (row + 1, col) in self.obstacles)
        obstacle_east  = int(col == self.grid_size - 1 or (row, col + 1) in self.obstacles)
        obstacle_west  = int(col == 0 or (row, col - 1) in self.obstacles)

        # passenger_look = 1 if passenger is in or adjacent to the taxi position
        passenger_look = self._adjacent_or_same((row, col), self.passenger_loc)
        # destination_look = 1 if destination is in or adjacent to the taxi position
        destination_look = self._adjacent_or_same((row, col), self.destination)

        obs = (
            row, col,
            stations_list[0][0], stations_list[0][1],
            stations_list[1][0], stations_list[1][1],
            stations_list[2][0], stations_list[2][1],
            stations_list[3][0], stations_list[3][1],
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            passenger_look, destination_look
        )
        return obs

    def _valid_position(self, r, c):
        """
        Check whether (r, c) is within grid and not an obstacle.
        """
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        if (r, c) in self.obstacles:
            return False
        return True

    def _adjacent_or_same(self, pos1, pos2):
        """
        Return 1 if pos2 is the same or immediately adjacent (manhattan distance 1 or 0).
        """
        return int(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) <= 1)

    def render_env(self, action=None, step=None):
        """
        Optional: You can print a textual representation of the grid with the taxi.
        This method is just for debugging; you can tailor it to your preferences.
        """
        # Building a simple grid representation for illustration
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # Mark stations
        # (R, G, Y, B) for corners, but we just place letters or you can do something else
        # This assumes you have exactly 4 stations in corners, as above
        station_labels = ['R', 'G', 'Y', 'B']
        for i, (sx, sy) in enumerate(self.stations):
            grid[sx][sy] = station_labels[i]

        # Mark obstacles
        for (ox, oy) in self.obstacles:
            grid[ox][oy] = 'X'

        # Mark taxi
        tx, ty = self.taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[tx][ty] = 'T'

        print(f"\nStep: {step}, Action: {action}, Fuel: {self.current_fuel}")
        for row in grid:
            print(" ".join(row))
        print()

def run_agent(agent_file, env_config=None, render=False):
    """
    Utility function to load your student_agent.py, create an instance of SimpleTaxiEnv,
    run an episode, and print out final reward and steps.
    """
    if env_config is None:
        env_config = {"grid_size": 5, "fuel_limit": 50}
    
    # Dynamically import the student's agent code
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    # Create environment
    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step_count = 0

    while not done:
        action = student_agent.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        total_reward += reward
        step_count += 1

        if render:
            env.render_env(action=action, step=step_count)

    print(f"Episode finished in {step_count} steps with total reward = {total_reward:.2f}")
    return total_reward

# If you run this file directly (python simple_custom_taxi_env.py),
# we can do a quick test with a random agent from student_agent.py
if __name__ == "__main__":
    # Example usage:
    final_score = run_agent("student_agent.py", env_config={"grid_size":8, "fuel_limit":2000}, render=True)
    print("Final Score from run_agent:", final_score)
