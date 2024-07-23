from customGrid import GridworldEnv

env = GridworldEnv()
obs = env.reset()
    
done = False
step_num = 1

# Run one episode
while not done:
    # Sample a random action from the action space
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    
    print(f"step#:{step_num} reward:{reward} done:{done} info:{info}")
        
    step_num += 1
    env.render()

env.close()