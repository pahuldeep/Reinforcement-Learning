import numpy as np
import copy
import gymnasium as gym

# Define constants
EMPTY = BLACK = 0
WALL = GRAY = 1
AGENT = BLUE = 2 
MINE = RED = 3
GOAL = GREEN = 4
SUCCESS = PINK = 5

COLOR_MAP = {
    BLACK: [0.0, 0.0, 0.0],
    GRAY: [0.5, 0.5, 0.5],
    BLUE: [0.0, 0.0, 1.0],
    RED: [1.0, 0.0, 0.0],
    GREEN: [0.0, 1.0, 0.0],
    PINK: [1.0, 0.0, 1.0],
}

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

class GridworldEnv(gym.Env):
    def __init__(self, max_steps=100):
        # Observations
        self.grid_layout = """  1 1 1 1 1 1 1 1
                                1 2 0 0 0 0 0 1
                                1 0 1 1 1 0 0 1
                                1 0 1 0 1 0 0 1
                                1 0 1 4 1 0 0 1
                                1 0 3 0 0 0 0 1
                                1 0 0 0 0 0 0 1
                                1 1 1 1 1 1 1 1 """
         
        self.initial_grid_state = np.fromstring(self.grid_layout, dtype=int, sep=" ")
        self.initial_grid_state = self.initial_grid_state.reshape(8, 8)
        self.grid_state = copy.deepcopy(self.initial_grid_state)

        self.observation_space = gym.spaces.Box(low=0, high=6, shape=self.grid_state.shape)
        self.img_shape = [256, 256, 3]
        self.metadata = {"render.modes": ["human"]}

        # Actions
        self.action_space = gym.spaces.Discrete(5)
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.action_pos_dict = {
            NOOP: [0, 0],
            UP: [-1, 0],
            DOWN: [1, 0],
            LEFT: [0, -1],
            RIGHT: [0, 1],
        }

        self.agent_state, self.agent_goal_state = self.get_state()

        # To keep track of number of steps
        self.step_num = 0 
        self.max_steps = max_steps
        self.done = False
        self.info = {"status": "Live"}
        self.viewer = None

    def step(self, action):
        """return next observation, reward, done, info"""
        action = int(action)
        reward = 0.0

        info = {"success": True}
        done = False
        
        next_obs = (self.agent_state[0] + self.action_pos_dict[action][0], self.agent_state[1] + self.action_pos_dict[action][1],)

        # Determine the reward
        if action == NOOP:
            return self.grid_state, reward, False, info
        
        next_state_valid = (next_obs[0] < 0 or next_obs[0] >= self.grid_state.shape[0]) or (next_obs[1] < 0 or next_obs[1] >= self.grid_state.shape[1])
        if next_state_valid:
            info['success'] = False
            return self.grid_state, reward, False, info
        
        next_state = self.grid_state[next_obs[0], next_obs[1]]

        if next_state == WALL:
            info['success'] = False
            reward = -0.1
            return self.grid_state, reward, False, info
        
        elif next_state == GOAL:
            done = True
            reward = 1
        
        elif next_state == MINE:
            done = True
            reward = -1

        self.grid_state[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(next_obs)
        self.grid_state[self.agent_state[0], self.agent_state[1]] = AGENT

        self.step_num += 1
        if self.step_num >= self.max_steps:
            done = True
        
        self.render("human")
        return self.grid_state, reward, done, info

    def reset(self):
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        self.agent_state, self.agent_goal_state = self.get_state()

        self.step_num = 0
        self.done = False
        self.info["status"] = "Live"

        return self.grid_state

    def get_state(self):
        start_state = np.where(self.grid_state == AGENT)
        goal_state = np.where(self.grid_state == GOAL)

        if not(start_state[0].size and goal_state[0].size):
            raise ValueError("Start and/or Goal state not present in the Grid_world")

        start_state = (start_state[0][0], start_state[1][0])
        goal_state = (goal_state[0][0], goal_state[1][0])

        return start_state, goal_state
    
    def gridarray_to_image(self, img_shape=None):
        if img_shape is None:
            img_shape = self.img_shape

        observation = np.zeros(img_shape)
        scale_x = int(observation.shape[0] / self.grid_state.shape[0])
        scale_y = int(observation.shape[1] / self.grid_state.shape[1])

        for i in range(self.grid_state.shape[0]):
            for j in range(self.grid_state.shape[1]):
                for k in range(3):  # 3-channel RGB image
                    pixel_value = COLOR_MAP[self.grid_state[i, j]][k]
                    observation[
                        i * scale_x : (i + 1) * scale_x,
                        j * scale_y : (j + 1) * scale_y,
                        k,
                    ] = pixel_value

        return (255 * observation).astype(np.uint8)
    
    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return 
        
        img = self.gridarray_to_image()

        if mode == "rgb_array":
            return img
        
        elif mode == "human":
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.axis('off')
            plt.show()

    def close(self):
        self.render(close=True)
