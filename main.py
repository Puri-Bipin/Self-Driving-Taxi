import numpy as np
import time
import pandas as pd
from src.environment import TaxiEnvironment
from src.policy_iteration import PolicyIteration
from src.animation import TaxiAnimator
from src.value_iteration import ValueIteration
from src.q_learning import QLearning


def run_policy_iteration(env):
    """Run Policy Iteration and return results"""
    
    print("=" * 50)
    print("POLICY ITERATION")
    print("=" * 50)
    
    start_time = time.time()
    pi_solver = PolicyIteration(env, gamma=0.9)
    pi_policy, pi_values = pi_solver.solve(max_iterations=500)
    pi_time = time.time() - start_time
    
    # Test policy performance
    avg_reward, avg_steps = test_policy(env, pi_policy)
    
    print(f"Computation Time: {pi_time:.2f}s")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return pi_policy, pi_time, avg_reward, avg_steps, {"policy": pi_policy, "values": pi_values}

def test_policy(env, policy, num_tests=10, max_steps=100):
    """Test a policy and return average performance"""
    total_reward = 0
    total_steps = 0
    
    for test in range(num_tests):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = np.argmax(policy[state])
            state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        total_reward += episode_reward
        total_steps += steps
    
    avg_reward = total_reward / num_tests
    avg_steps = total_steps / num_tests
    
    return avg_reward, avg_steps

def run_value_iteration(env):
    """Run Value Iteration and return results"""
    
    print("\n" + "=" * 50)
    print("VALUE ITERATION")
    print("=" * 50)
    
    start_time = time.time()
    vi_solver = ValueIteration(env, gamma=0.9)
    vi_policy, vi_values = vi_solver.solve(max_iterations=1000)
    vi_time = time.time() - start_time
    
    # Test policy performance
    avg_reward, avg_steps = test_policy(env, vi_policy)
    
    print(f"Computation Time: {vi_time:.2f}s")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return vi_policy, vi_time, avg_reward, avg_steps, {"policy": vi_policy, "values": vi_values}

def run_q_learning(env, num_episodes=5000):
    """Run Q-Learning and return results"""
    
    print("\n" + "=" * 50)
    print("Q-LEARNING")
    print("=" * 50)
    
    start_time = time.time()
    q_agent = QLearning(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    rewards_history, steps_history = q_agent.train(num_episodes=num_episodes, print_every=500)
    q_time = time.time() - start_time
    
    # Extract policy from Q-table
    q_policy = q_agent.get_policy()
    
    # Test policy performance
    avg_reward, avg_steps = test_policy(env, q_policy)
    
    print(f"Computation Time: {q_time:.2f}s")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return q_policy, q_time, avg_reward, avg_steps, {"policy": q_policy, "q_table": q_agent.Q, "rewards_history": rewards_history}

def create_animations(env, results):
    """Create animations for all algorithms"""
    print("\n" + "=" * 50)
    print("CREATING ANIMATIONS")
    print("=" * 50)
    
    animations_created = []
    for algo_name, algo_data in results.items():
        if algo_name != "comparison":
            print(f"\nCreating animation for {algo_name}...")
            try:
                animator = TaxiAnimator(env)
                animator.create_animation(algo_data["policy"], algo_name)
                animations_created.append(algo_name)
                print(f"✓ {algo_name} animation created!")
            except Exception as e:
                print(f"✗ {algo_name} animation failed: {e}")
    
    return animations_created

def display_comparison_table(results):
    """Display a comparison table of all algorithms"""
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    
    # Create comparison data
    comparison_data = []
    
    for algo_name, algo_data in results.items():
        if algo_name != "comparison":
            comparison_data.append({
                "Algorithm": algo_name,
                "Time (s)": f"{algo_data['time']:.2f}",
                "Avg Reward": f"{algo_data['avg_reward']:.1f}",
                "Avg Steps": f"{algo_data['avg_steps']:.1f}",
                "Method": algo_data["method"],
                "Model-Based": algo_data["model_based"],
                "Convergence": algo_data["convergence"]
            })
    
    # Create DataFrame for nice formatting
    df = pd.DataFrame(comparison_data)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    
    print("\n" + df.to_string(index=False))
    
    # Add some analysis
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)
    
    # Find best in each category
    algorithms = [row["Algorithm"] for row in comparison_data]
    times = [float(row["Time (s)"]) for row in comparison_data]
    rewards = [float(row["Avg Reward"]) for row in comparison_data]
    steps = [float(row["Avg Steps"]) for row in comparison_data]
    
    print(f"Fastest Algorithm: {algorithms[times.index(min(times))]} ({min(times):.2f}s)")
    print(f"Highest Reward: {algorithms[rewards.index(max(rewards))]} ({max(rewards):.1f})")
    print(f"Most Efficient: {algorithms[steps.index(min(steps))]} ({min(steps):.1f} steps)")
    
    return df

def create_algorithm_summary():
    """Create a summary of algorithm characteristics"""
    print("\n" + "=" * 70)
    print("ALGORITHM CHARACTERISTICS")
    print("=" * 70)
    
    summary = """
    Key Differences:
    
    1. POLICY ITERATION
       - Alternates between policy evaluation and improvement
       - Guaranteed convergence to optimal policy
       - Model-based (requires transition probabilities)
       - Slower but very stable
    
    2. VALUE ITERATION
       - Directly updates value function using Bellman optimality
       - More computationally efficient than policy iteration
       - Model-based (requires transition probabilities)
       - Often converges faster
    
    3. Q-LEARNING
       - Model-free (learns from experience)
       - Uses exploration-exploitation tradeoff (ε-greedy)
       - Suitable for unknown environments
       - Requires more samples but no model needed
    
    Performance Notes:
    - Model-based methods (PI, VI) are typically faster with known models
    - Q-Learning is more flexible for unknown environments
    - All should converge to similar optimal policies in deterministic environments
    """
    
    print(summary)

def main():
    # Create environment
    env = TaxiEnvironment(
        grid_size=6,
        obstacles=[(1,1), (1,3), (2,4), (3,1), (3,4), (4,4)],
        taxi_start=(2, 0),
        passenger_start=(0, 5),
        destination=(5, 2)
    )
    
    print("\n" + "=" * 70)
    print("SELF-DRIVING TAXI - RL ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    print(f"Obstacles: {len(env.obstacles)}")
    print(f"Taxi Start: {env.taxi_start}")
    print(f"Passenger Start: {env.passenger_start}")
    print(f"Destination: {env.destination}")
    
    # Dictionary to store all results
    results = {}
    
    print("\n" + "=" * 70)
    print("RUNNING ALL ALGORITHMS")
    print("=" * 70)
    
    # Run Policy Iteration
    print("\n>>> Running Policy Iteration...")
    pi_policy, pi_time, pi_reward, pi_steps, pi_data = run_policy_iteration(env)
    results["Policy Iteration"] = {
        "policy": pi_policy,
        "time": pi_time,
        "avg_reward": pi_reward,
        "avg_steps": pi_steps,
        "method": "Policy Evaluation + Improvement",
        "model_based": "Yes",
        "convergence": "Guaranteed",
        "data": pi_data
    }
    
    # Run Value Iteration
    print("\n>>> Running Value Iteration...")
    vi_policy, vi_time, vi_reward, vi_steps, vi_data = run_value_iteration(env)
    results["Value Iteration"] = {
        "policy": vi_policy,
        "time": vi_time,
        "avg_reward": vi_reward,
        "avg_steps": vi_steps,
        "method": "Bellman Optimality Updates",
        "model_based": "Yes",
        "convergence": "Guaranteed",
        "data": vi_data
    }
    
    # Run Q-Learning
    print("\n>>> Running Q-Learning...")
    q_policy, q_time, q_reward, q_steps, q_data = run_q_learning(env, num_episodes=5000)
    results["Q-Learning"] = {
        "policy": q_policy,
        "time": q_time,
        "avg_reward": q_reward,
        "avg_steps": q_steps,
        "method": "Temporal Difference Learning",
        "model_based": "No",
        "convergence": "Probabilistic",
        "data": q_data
    }
    
    # Create animations
    animations_created = create_animations(env, results)
    
    # Display comparison table
    comparison_df = display_comparison_table(results)
    
    # Add algorithm summary
    create_algorithm_summary()
    
    # Policy comparison
    print("\n" + "=" * 70)
    print("POLICY CONSISTENCY CHECK")
    print("=" * 70)
    
    # Check how similar the policies are
    print("\nPolicy Agreement (percentage of states with same optimal action):")
    algorithms = list(results.keys())
    
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1 = algorithms[i]
            algo2 = algorithms[j]
            policy1 = results[algo1]["policy"]
            policy2 = results[algo2]["policy"]
            
            # Compare policies
            agreements = 0
            for s in range(env.n_states):
                if np.argmax(policy1[s]) == np.argmax(policy2[s]):
                    agreements += 1
            
            agreement_percent = (agreements / env.n_states) * 100
            print(f"  {algo1} vs {algo2}: {agreement_percent:.1f}% agreement")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    
    if animations_created:
        print(f"\n✓ Created animations for: {', '.join(animations_created)}")
    
    print("\nSummary of results has been saved in memory.")
    print("Check the generated animation files to see each algorithm in action!")
    
    # Return results for further analysis if needed
    return results, comparison_df

if __name__ == "__main__":
    results, comparison_df = main()