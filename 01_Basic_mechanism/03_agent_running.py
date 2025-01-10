import random

class Envirnoment:
    def __init__(self):
        self.total_steps = 10
        
    def get_observation(self):
        return [ 1, 2, 3, 4, 5 ]
    
    def done(self):
        return self.total_steps == 0
    
    def action(self, action):
        self.total_steps -= 1

        action_reward = action
        return action_reward
    
class Agent:
    def __init__(self,):
        self.reward = 0

    def step(self, env):
        actions = env.get_observation()
        reward = env.action(random.choice(actions))     # doing in envirnoment  
        self.reward += reward
        return reward


if __name__ == "__main__":

    env = Envirnoment()
    agent = Agent()

    while not env.done():
        print("observed moves:", agent.step(env))

    print("Total reward got: %.4f" % agent.reward)

