import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py


gym.register_envs(ale_py)

model_path = "dqn_model.zip"
model = DQN.load(model_path)

env_id = "ALE/SpaceInvaders-v5"
env = gym.make(env_id, render_mode="human")
env.metadata['render_fps'] = 120  
env = AtariWrapper(env)

obs, _ = env.reset()

for episode in range(3):
    done = False
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0
    
    print(f"\nStarting Episode {episode + 1}...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        
        env.render()  
        
        if step_count % 100 == 0: 
            print(f"  Step {step_count} Reward: {total_reward:.1f}")
    
    print(f"Episode {episode + 1} completed!")
    print(f" Action: {action}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Steps taken: {step_count}")

env.close()
print("\nGame finished!")