import gymnasium as gym

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")

obs = env.reset()

for step_num in range(10):
    env.render()
    env.step(env.action_space.sample()) 
    print(f"#step: {step_num}")