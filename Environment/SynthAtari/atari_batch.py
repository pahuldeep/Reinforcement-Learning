import cv2
import random
import numpy as np

import torch
import gymnasium as gym
from gymnasium import spaces

BATCH_SIZE = 16
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
        new_obs = np.moveaxis(new_obs, 2, 0)                            # transform (w, h, c) -> (c, w, h)

        return new_obs.astype(np.float32)

def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        action = e.action_space.sample()
        obs, reward, is_done, is_trunc, _ = e.step(action)
        
        if np.mean(obs) > 0.01:
            batch.append(obs)

        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)    # Normalising input
            batch.clear()

        if is_done or is_trunc:
            e.reset()
