import gymnasium as gym
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v1', render_mode = "rgb_array")
env.reset()

env.step(1)

plt.imshow(env.render())
plt.show()
