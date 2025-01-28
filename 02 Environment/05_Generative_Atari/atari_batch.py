import random
import numpy as np

import torch

BATCH_SIZE = 16

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
