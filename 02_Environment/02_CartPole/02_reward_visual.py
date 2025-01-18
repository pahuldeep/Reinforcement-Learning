import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
obs = env.reset()
total_reward = 0
total_step = 0

while total_step < 100:
    
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) 
    print(f"step#:{total_step} reward#:{reward} observation:{obs} state:{env.action_space.sample()}")
    
    total_reward += reward
    total_step += 1

    env.render()


print("Reward: ", total_reward)