import numpy as np
import time
from src.environment import TaxiEnvironment
from src.policy_iteration import PolicyIteration
from src.animation import TaxiAnimator

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
    total_reward = 0
    total_steps = 0
    num_tests = 10
    
    for test in range(num_tests):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(100):
            action = np.argmax(pi_policy[state])
            state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        total_reward += episode_reward
        total_steps += steps
    
    avg_reward = total_reward / num_tests
    avg_steps = total_steps / num_tests
    
    print(f"Computation Time: {pi_time:.2f}s")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return pi_policy, pi_time, avg_reward, avg_steps

def main():
    # Create environment
    env = TaxiEnvironment(
        grid_size=6,
        obstacles=[(1,1), (1,3), (2,4), (3,1), (3,4), (4,4)],
        taxi_start=(2, 0),
        passenger_start=(0, 5),
        destination=(5, 2)
    )
    
    print("POLICY ITERATION DEMONSTRATION")
    print(f"Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    print()
    
    # Run Policy Iteration
    policy, comp_time, avg_reward, avg_steps = run_policy_iteration(env)
    
    # Create animation
    print("\n" + "=" * 50)
    print("CREATING ANIMATION")
    print("=" * 50)
    
    print("Creating animation for Policy Iteration...")
    try:
        animator = TaxiAnimator(env)
        animator.create_animation(policy, "Policy Iteration")
        print("Animation created successfully!")
    except Exception as e:
        print(f"Animation failed: {e}")
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()