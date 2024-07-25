import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
obs = env.reset()

for step_num in range(100): 
    env.render()
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"step#:{step_num} reward:{reward} observation: {obs} done:{terminated}")

    # Check if the episode has ended
    if terminated or truncated:
        obs = env.reset()

env.close()



