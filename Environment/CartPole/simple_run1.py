import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
obs = env.reset()

for step_num in range(100):
    env.render()
    env.step(env.action_space.sample()) 
    print(f"#step: {step_num}")
    