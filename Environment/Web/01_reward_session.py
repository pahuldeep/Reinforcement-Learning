import gymnasium as gym

import miniwob
gym.register_envs(miniwob)

env = gym.make("miniwob/click-test-v1", render_mode='human')
observation, info = env.reset()

total_reward = 0
total_steps = 0

while True:
    # Random action (useful for exploration in MiniWoB)
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    
    # Update rewards and steps
    total_reward += reward
    total_steps += 1
    
    if done or truncated:
        break

print(f"Episode done in steps: {total_steps}, total reward: {total_reward:.2f}")
env.close()
