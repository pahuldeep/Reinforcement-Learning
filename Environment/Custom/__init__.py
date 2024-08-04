from gymnasium.envs.registration import register

register(
    id="Gridworld-v0", entry_point="custom_env.customGrid:GridworldEnv",
)