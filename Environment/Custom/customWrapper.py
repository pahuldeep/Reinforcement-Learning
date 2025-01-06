import gymnasium as gym
import random

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, randomness=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.randomness = randomness
    
    def action(self, action):
        if random.random() < self.randomness:
            action = self.env.action_space.sample()
            print(f"Random action taken: {action}")
        return action 

if __name__ == "__main__":
    
    env = RandomActionWrapper(gym.make("CartPole-v1"))

    observation, info = env.reset()
    total_reward = 0

    while True:

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break
    
    print(f"Total Reward: {total_reward}")
