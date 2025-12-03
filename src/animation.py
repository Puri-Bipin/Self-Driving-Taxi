import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from src.environment import TaxiEnvironment, Action

class TaxiAnimator:
    def __init__(self, env):
        self.env = env
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
    def create_animation(self, policy, algorithm_name, max_steps=50):
        """Create animation of taxi following policy"""
        state = self.env.reset()
        states_history = [self.env.get_state()]
        actions_history = []
        rewards_history = []
        
        # Run simulation to collect data
        for step in range(max_steps):
            action = np.argmax(policy[state])
            state, reward, done = self.env.step(action)
            
            states_history.append(state)
            actions_history.append(action)
            rewards_history.append(reward)
            
            if done:
                break
        
        # Create animation
        self.ax.clear()
        self.ax.set_title(f"Self-Driving Taxi - {algorithm_name}\nStep: 0, Total Reward: 0")
        self.ax.set_xticks(np.arange(-0.5, self.env.grid_size, 1))
        self.ax.set_yticks(np.arange(-0.5, self.env.grid_size, 1))
        self.ax.grid(True)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Create initial frame
        im = self._draw_frame(0, states_history, actions_history, rewards_history, algorithm_name)
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, 
            self._update_frame,
            frames=len(states_history),
            fargs=(states_history, actions_history, rewards_history, algorithm_name),
            interval=800,  # 0.8 seconds between frames
            repeat=False
        )
        
        # Save animation
        anim.save(f'results/{algorithm_name.lower().replace(" ", "_")}_animation.gif', 
                 writer='pillow', fps=1.5)
        
        plt.show()
        return anim
    
    def _draw_frame(self, frame_idx, states_history, actions_history, rewards_history, algorithm_name):
        """Draw a single frame"""
        self.ax.clear()
        
        # Create grid background
        grid = np.zeros((self.env.grid_size, self.env.grid_size))
        
        # Draw obstacles
        for obs in self.env.obstacles:
            grid[obs[0]][obs[1]] = -1
        
        # Draw special locations
        grid[self.env.passenger_start[0]][self.env.passenger_start[1]] = 2
        grid[self.env.destination[0]][self.env.destination[1]] = 3
        
        # Draw current state
        current_state = states_history[frame_idx]
        taxi_pos, passenger_status = self.env.decode_state(current_state)
        
        # Draw taxi
        if passenger_status == 1:  # Passenger in taxi
            grid[taxi_pos[0]][taxi_pos[1]] = 4
        else:  # Taxi alone
            grid[taxi_pos[0]][taxi_pos[1]] = 1
        
        # Create colormap
        cmap = plt.cm.Set3
        im = self.ax.imshow(grid, cmap=cmap, vmin=-1, vmax=4)
        
        # Add text annotations
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                if (i, j) in self.env.obstacles:
                    # Changed from '🚧' to 'X'
                    self.ax.text(j, i, 'X', ha='center', va='center', fontsize=20, 
                               fontweight='bold', color='red')
                elif (i, j) == self.env.passenger_start:
                    # Changed from '👤' to 'P'
                    self.ax.text(j, i, 'P', ha='center', va='center', fontsize=20, 
                               fontweight='bold', color='blue')
                elif (i, j) == self.env.destination:
                    # Changed from '🏁' to 'D'
                    self.ax.text(j, i, 'D', ha='center', va='center', fontsize=20, 
                               fontweight='bold', color='green')
                elif (i, j) == taxi_pos:
                    if passenger_status == 1:
                        # Changed from '🚕👤' to 'T+P'
                        self.ax.text(j, i, 'T+P', ha='center', va='center', fontsize=15, 
                                   fontweight='bold', color='purple')
                    else:
                        # Changed from '🚕' to 'T'
                        self.ax.text(j, i, 'T', ha='center', va='center', fontsize=20, 
                                   fontweight='bold', color='orange')
        
        # Add step info
        total_reward = sum(rewards_history[:frame_idx])
        if frame_idx > 0:
            last_action = Action(actions_history[frame_idx-1]).name
            action_text = f"Last Action: {last_action}"
        else:
            action_text = "Initial State"
            
        self.ax.set_title(
            f"Self-Driving Taxi - {algorithm_name}\n"
            f"Step: {frame_idx}, Total Reward: {total_reward}\n"
            f"{action_text}\n"
            f"Passenger: {'Waiting' if passenger_status == 0 else 'In Taxi' if passenger_status == 1 else 'Delivered'}"
        )
        
        return im
    
    def _update_frame(self, frame_idx, states_history, actions_history, rewards_history, algorithm_name):
        """Update animation frame"""
        return self._draw_frame(frame_idx, states_history, actions_history, rewards_history, algorithm_name)