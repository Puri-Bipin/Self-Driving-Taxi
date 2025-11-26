import numpy as np
from src.environment import TaxiEnvironment

class ValueIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # Initialize value function
        self.V = np.zeros(self.n_states)
    
    def value_iteration(self):
        """Perform value iteration to find optimal value function"""
        iteration = 0
        while True:
            delta = 0
            iteration += 1
            
            # Update each state
            for s in range(self.n_states):
                v = self.V[s]
                
                # Compute value for each action
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for prob, next_s, reward, _ in self.env.get_transition_probabilities(s, a):
                        action_values[a] += prob * (reward + self.gamma * self.V[next_s])
                
                # Update value with maximum over actions (Bellman optimality equation)
                self.V[s] = np.max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            print(f"Value Iteration - Iteration {iteration}, Delta: {delta:.6f}")
            
            # Check convergence
            if delta < self.theta:
                print(f"Value function converged after {iteration} iterations")
                break
        
        return iteration
    
    def extract_policy(self):
        """Extract optimal policy from value function"""
        policy = np.zeros([self.n_states, self.n_actions])
        
        for s in range(self.n_states):
            # Compute value for each action
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_s, reward, _ in self.env.get_transition_probabilities(s, a):
                    action_values[a] += prob * (reward + self.gamma * self.V[next_s])
            
            # Set policy to be greedy with respect to value function
            best_action = np.argmax(action_values)
            policy[s] = np.eye(self.n_actions)[best_action]
        
        return policy
    
    def solve(self, max_iterations=1000):
        """Run value iteration and extract policy"""
        print("Starting Value Iteration...")
        
        # Run value iteration
        iterations = self.value_iteration()
        
        # Extract optimal policy
        policy = self.extract_policy()
        
        print(f"Policy extracted successfully")
        
        return policy, self.V