import gymnasium as gym

env = gym.make('phys2d/CartPole-v1', render_mode="human")
obs = env.reset()

for step_num in range(500):
    env.render()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample()) 
    print(f"step#:{step_num} observation:{obs} state:{env.action_space.sample()}")