import numpy as np
from src.environment import TaxiEnvironment

class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # Initialize policy and value function
        self.policy = np.ones([self.n_states, self.n_actions]) / self.n_actions
        self.V = np.zeros(self.n_states)
    
    def policy_evaluation(self):
        """Evaluate the current policy"""
        while True:
            delta = 0
            for s in range(self.n_states):
                v = self.V[s]
                new_v = 0
                
                # Sum over all actions for this state
                for a, action_prob in enumerate(self.policy[s]):
                    # Sum over all possible next states
                    for prob, next_s, reward, _ in self.env.get_transition_probabilities(s, a):
                        new_v += action_prob * prob * (reward + self.gamma * self.V[next_s])
                
                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))
            
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        """Improve the policy based on current value function"""
        policy_stable = True
        
        for s in range(self.n_states):
            old_action = np.argmax(self.policy[s])
            
            # Find best action for this state
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_s, reward, _ in self.env.get_transition_probabilities(s, a):
                    action_values[a] += prob * (reward + self.gamma * self.V[next_s])
            
            # Update policy to be greedy
            best_action = np.argmax(action_values)
            self.policy[s] = np.eye(self.n_actions)[best_action]
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def solve(self, max_iterations=100):
        """Run policy iteration until convergence"""
        for i in range(max_iterations):
            print(f"Policy Iteration - Iteration {i+1}")
            
            # Policy Evaluation
            self.policy_evaluation()
            
            # Policy Improvement
            policy_stable = self.policy_improvement()
            
            if policy_stable:
                print(f"Policy converged after {i+1} iterations")
                break
        
        return self.policy, self.V