import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
obs = env.reset()
total_reward = 0

# random actions with maximum reward
for step_num in range(100): 

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward    
    
    if terminated or truncated: # Check if the episode has ended
        obs = env.reset()

    print(f"step#:{step_num} observation: {obs} done:{terminated} Info: {info}")
        
    env.render()
env.close()

print("Reward: ", total_reward)