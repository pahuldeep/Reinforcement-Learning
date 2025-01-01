import random

class Envirnoment:
    def __init__(self):
        self.total_steps = 10
        
    def get_actions(self):
        return [ 0, 1 ]
    
    def done(self):
        return self.total_steps == 0
    
    def action(self, action: int):
        self.total_steps -= 1

        action_reward = random.random()
        return action_reward
    
class Agent:
    def __init__(self,):
        self.reward = 0

    def step(self, env: Envirnoment):
        actions = env.get_actions()                 # get from envirnoment
        reward = env.action(random.choice(actions)) # doing in envirnoment  
        self.reward += reward
        return actions


if __name__ == "__main__":

    env = Envirnoment()
    agent = Agent()

    while not env.done():
        print("agents move:", agent.step(env))

    print("Total reward got: %.4f" % agent.reward)

