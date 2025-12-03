# Self-Driving Taxi RL Comparison
A reinforcement learning project that compares three RL algorithms (Policy Iteration, Value Iteration, and Q-Learning) for a self-driving taxi simulation with visual animations.

# Algorithms Compared
- Policy Iteration: Model-based algorithm using policy evaluation and improvement cycles
- Value Iteration: Model-based algorithm using Bellman optimality updates
- Q-Learning: Model-free temporal difference learning with exploration-exploitation tradeoff

# Quick Start
### 1. Install Required Libraries
  ```bash
   pip install numpy matplotlib pandas
```
### 2. Project Structure
    
    Self-Driving-Taxi/
    ├── src/
    │   ├── environment.py      # Taxi environment implementation
    │   ├── policy_iteration.py # Policy Iteration algorithm
    │   ├── value_iteration.py  # Value Iteration algorithm
    │   ├── q_learning.py       # Q-Learning algorithm
    │   └── animation.py        # Animation visualization
    ├── results/                # Generated animations (created automatically)
    ├── main.py                 # Main script to run comparisons
    └── README.md               # This file
    
### 3. Run the Project
```bash
cd Self-Driving-Taxi
python main.py
```
    
### 4. What Happens
- All three algorithms run sequentially
- Performance metrics are displayed in a comparison table
- Animations are saved as GIFs in the results/ folder
- Policy consistency is checked across algorithms

# Output
- Console: Performance comparison table showing computation time, average reward, and steps
- Animations: Three GIF files (one per algorithm) showing taxi movement

# Key Features
- 6×6 grid with obstacles
- Visual animation of taxi pathfinding
- Comprehensive algorithm comparison
- Model-based vs model-free RL demonstration
- Deterministic environment with clear state transitions

# Configuration
- Edit main.py to change:
- Grid size and obstacles
- Algorithm parameters (gamma, epsilon, learning rate)
- Training episodes for Q-Learning

# Results
After running, check the results/ folder for:

- policy_iteration_animation.gif
- value_iteration_animation.gif
- q_learning_animation.gif


