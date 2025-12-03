import numpy as np
from src.environment import TaxiEnvironment

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-Learning Algorithm
        
        Args:
            env: TaxiEnvironment instance
            alpha: Learning rate (how much we update Q-values)
            gamma: Discount factor (importance of future rewards)
            epsilon: Exploration rate (probability of random action)
        """
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # Initialize Q-table with zeros
        self.Q = np.zeros([self.n_states, self.n_actions])
    
    def get_action(self, state, training=True):
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.Q[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Get maximum Q-value for next state
        max_next_q = np.max(self.Q[next_state])
        
        # Q-Learning update
        current_q = self.Q[state, action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state, action] = new_q
    
    def train_episode(self, max_steps=100):
        """
        Train for one episode
        
        Returns:
            total_reward: Total reward earned in episode
            steps: Number of steps taken
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose action using epsilon-greedy
            action = self.get_action(state, training=True)
            
            # Take action and observe result
            next_state, reward, done = self.env.step(action)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state)
            
            # Update state and tracking variables
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
    
    def train(self, num_episodes=1000, print_every=100):
        """
        Train Q-Learning agent for multiple episodes
        
        Args:
            num_episodes: Number of training episodes
            print_every: Print progress every N episodes
        
        Returns:
            rewards_history: List of rewards per episode
            steps_history: List of steps per episode
        """
        rewards_history = []
        steps_history = []
        
        print(f"Training Q-Learning for {num_episodes} episodes...")
        print(f"Parameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
        print("-" * 60)
        
        for episode in range(num_episodes):
            # Train one episode
            episode_reward, episode_steps = self.train_episode()
            
            rewards_history.append(episode_reward)
            steps_history.append(episode_steps)
            
            # Print progress
            if (episode + 1) % print_every == 0:
                avg_reward = np.mean(rewards_history[-print_every:])
                avg_steps = np.mean(steps_history[-print_every:])
                print(f"Episode {episode+1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
        
        print("-" * 60)
        print(f"Training complete!")
        print(f"Final 100 episodes - Avg Reward: {np.mean(rewards_history[-100:]):.2f}, "
              f"Avg Steps: {np.mean(steps_history[-100:]):.2f}")
        
        return rewards_history, steps_history
    
    def get_policy(self):
        """
        Extract deterministic policy from Q-table
        Policy is greedy with respect to Q-values
        
        Returns:
            policy: Array of shape [n_states, n_actions] with one-hot encoding
        """
        policy = np.zeros([self.n_states, self.n_actions])
        
        for s in range(self.n_states):
            best_action = np.argmax(self.Q[s])
            policy[s, best_action] = 1.0
        
        return policy
    
    def test_policy(self, num_tests=10, max_steps=100):
        """
        Test the learned policy
        
        Returns:
            avg_reward: Average reward over test episodes
            avg_steps: Average steps over test episodes
        """
        total_reward = 0
        total_steps = 0
        
        for test in range(num_tests):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Use greedy action (no exploration)
                action = self.get_action(state, training=False)
                state, reward, done = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            total_reward += episode_reward
            total_steps += steps
        
        avg_reward = total_reward / num_tests
        avg_steps = total_steps / num_tests
        
        return avg_reward, avg_steps