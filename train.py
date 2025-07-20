import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
import ale_py

# Register Atari environments from ALE
gym.register_envs(ale_py)

def create_environment():
    """
    Create the Atari Space Invaders environment with proper preprocessing.
    """
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    env = AtariWrapper(env)
    return env

def compare_policies():
    """
    Train and evaluate both CnnPolicy and MlpPolicy
    """
    results = {}
    
    for policy_name in ["CnnPolicy", "MlpPolicy"]:
        env = create_environment()
        
        # Define DQN model with consistent hyperparameters
        model = DQN(
            policy=policy_name,
            env=env,
            learning_rate=2e-4,
            gamma=0.995,
            batch_size=64,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            exploration_fraction=0.15,
            verbose=0  # Suppress training logs
        )
        
        # Short training run for comparison
        model.learn(total_timesteps=300_000)
        
        # Evaluate the policy
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        results[policy_name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }
        
        # Save the trained model
        model.save(f"dqn_{policy_name.lower()}_model.zip")
        env.close()
    
    # Determine the winning policy based on higher mean reward
    cnn_score = results["CnnPolicy"]["mean_reward"]
    mlp_score = results["MlpPolicy"]["mean_reward"]
    winner = "CnnPolicy" if cnn_score > mlp_score else "MlpPolicy"
    
    return results, winner

def train_best_model(best_policy):
    """
    Train the winning policy for a longer period and return final performance.
    """
    env = create_environment()

    model = DQN(
        policy=best_policy,
        env=env,
        learning_rate=2e-4,
        gamma=0.995,
        batch_size=64,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        exploration_fraction=0.15,
        verbose=0
    )

    # Train longer for better performance
    model.learn(total_timesteps=500_000)
    model.save("dqn_model.zip")

    # Evaluate after training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    env.close()
    return mean_reward, std_reward

def main():
    """
    Main pipeline to:
        1. Compare MLP vs CNN policies
        2. Select and train the best policy longer
        3. Output performance summary
    """
    policy_results, winning_policy = compare_policies()
    final_mean, final_std = train_best_model(winning_policy)
    
    print("Policy Comparison Results:")
    for policy, result in policy_results.items():
        print(f"  {policy}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    print(f"\nWinner: {winning_policy}")
    print(f"Final {winning_policy} model performance: {final_mean:.2f} ± {final_std:.2f}")

if __name__ == "__main__":
    main()
