import cv2
import numpy as np

import gymnasium as gym
from gymnasium import spaces

IMAGE_SIZE = 64

class InputWrapper(gym.ObservationWrapper):

    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(old_space.low), self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation):

        new_obs = cv2.resize( observation, (IMAGE_SIZE, IMAGE_SIZE))
        new_obs = np.moveaxis(new_obs, 2, 0)    # transform (w, h, c) -> (c, w, h)

        return new_obs.astype(np.float32)