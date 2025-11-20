import numpy as np
from enum import Enum

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    PICKUP = 4
    DROPOFF = 5

class TaxiEnvironment:
    def __init__(self, grid_size=5, obstacles=None, taxi_start=None, passenger_start=None, destination=None):
        self.grid_size = grid_size
        
        # Define special locations with custom values or defaults
        self.passenger_start = passenger_start if passenger_start is not None else (0, 0)
        self.destination = destination if destination is not None else (4, 4)
        self.taxi_start = taxi_start if taxi_start is not None else (0, 4)
        
        # Define obstacles (walls) - use custom or default
        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = [(1, 1), (1, 3), (3, 1), (3, 3)]
        
        # State: (taxi_row, taxi_col, passenger_location, destination)
        # passenger_location: 0=at start, 1=in taxi, 2=at destination
        self.n_states = grid_size * grid_size * 3  # *3 for passenger status
        
        self.n_actions = len(Action)
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.taxi_position = self.taxi_start
        self.passenger_in_taxi = False
        self.passenger_delivered = False
        return self.get_state()
    
    def get_state(self):
        """Convert current environment to state index"""
        taxi_row, taxi_col = self.taxi_position
        passenger_status = 1 if self.passenger_in_taxi else (0 if not self.passenger_delivered else 2)
        
        state = taxi_row * self.grid_size * 3 + taxi_col * 3 + passenger_status
        return state
    
    def decode_state(self, state):
        """Convert state index back to environment components"""
        passenger_status = state % 3
        remainder = state // 3
        taxi_col = remainder % self.grid_size
        taxi_row = remainder // self.grid_size
        return (taxi_row, taxi_col), passenger_status
    
    def step(self, action):
        """Take an action and return next_state, reward, done"""
        reward = -1  # Default step penalty
        done = False
        
        if action == Action.UP.value:
            new_pos = (max(0, self.taxi_position[0] - 1), self.taxi_position[1])
            if new_pos not in self.obstacles:
                self.taxi_position = new_pos
                
        elif action == Action.RIGHT.value:
            new_pos = (self.taxi_position[0], min(self.grid_size - 1, self.taxi_position[1] + 1))
            if new_pos not in self.obstacles:
                self.taxi_position = new_pos
                
        elif action == Action.DOWN.value:
            new_pos = (min(self.grid_size - 1, self.taxi_position[0] + 1), self.taxi_position[1])
            if new_pos not in self.obstacles:
                self.taxi_position = new_pos
                
        elif action == Action.LEFT.value:
            new_pos = (self.taxi_position[0], max(0, self.taxi_position[1] - 1))
            if new_pos not in self.obstacles:
                self.taxi_position = new_pos
                
        elif action == Action.PICKUP.value:
            if self.taxi_position == self.passenger_start and not self.passenger_in_taxi and not self.passenger_delivered:
                self.passenger_in_taxi = True
                reward = 5  # Good pickup reward
            else:
                reward = -10  # Illegal pickup
                
        elif action == Action.DROPOFF.value:
            if self.taxi_position == self.destination and self.passenger_in_taxi:
                self.passenger_in_taxi = False
                self.passenger_delivered = True
                reward = 20  # Successful delivery
                done = True
            else:
                reward = -10  # Illegal dropoff
        
        return self.get_state(), reward, done
    
    def get_transition_probabilities(self, state, action):
        """Get all possible next states and their probabilities"""
        # Save current state
        current_state = self.get_state()
        current_taxi_pos, current_passenger_status = self.decode_state(state)
        
        # Restore environment to the given state
        self.taxi_position = current_taxi_pos
        self.passenger_in_taxi = (current_passenger_status == 1)
        self.passenger_delivered = (current_passenger_status == 2)
        
        # In this deterministic environment, there's only one next state
        next_state, reward, done = self.step(action)
        
        # Restore original state
        self.taxi_position, _ = self.decode_state(current_state)
        
        return [(1.0, next_state, reward, done)]
    
    def render(self):
        """Visualize the current state"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = '█'
        
        # Mark special locations
        grid[self.passenger_start[0]][self.passenger_start[1]] = 'P'
        grid[self.destination[0]][self.destination[1]] = 'D'
        
        # Mark taxi
        taxi_char = 'T' if not self.passenger_in_taxi else 'T+P'
        grid[self.taxi_position[0]][self.taxi_position[1]] = taxi_char
        
        print("Grid:")
        for row in grid:
            print(' '.join(row))
        print(f"Passenger in taxi: {self.passenger_in_taxi}")
        print(f"Passenger delivered: {self.passenger_delivered}")
        print("-" * 20)