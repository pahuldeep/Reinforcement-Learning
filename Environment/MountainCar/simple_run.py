import gymnasium as gym

env = gym.make("MountainCarContinuous-v0", render_mode="human")
observation = env.reset()

total_reward = 0
for step_num in range(100):
    actions = env.action_space.sample()
    
    observation, reward, terminate, info, done = env.step(actions) 
    total_reward += reward
    print(f"#step: {step_num} Observation: {observation} Reward: {reward}" )

print(f"Total Reward: {total_reward}")


    