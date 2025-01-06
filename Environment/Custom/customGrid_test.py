from customGrid import GridworldEnv

env = GridworldEnv()
obs = env.reset()
    
done = False
step_num = 0
total_reward = 0

# Run one episode
while not done:

    action = env.action_space.sample()
    next_observation, reward, done, info = env.step(action)
    
    print(f"step#:{step_num} reward:{reward} done:{done} info:{info}")
    total_reward += reward
    step_num += 1
    env.render(mode="rgb_array")
    
print("Reward: ",total_reward)
env.close()
