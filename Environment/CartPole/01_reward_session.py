import gymnasium as gym

env = gym.make("CartPole-v1")

total_reward = 0
total_step = 0

observe = env.reset()

while True:
    actions = env.action_space.sample()
    observe, reward, is_done, is_trunc, _ = env.step(actions)
    
    total_reward += reward
    total_step += 1

    if is_done:
        break

print("Episode done in steps: %d, reward: %.2f"%(total_step, total_reward))