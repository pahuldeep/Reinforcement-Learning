def evaluate(agent, env, render=True):

    observation = env.reset()
    step_num = 0
    episode_reward = 0.0
    done = False

    while not done:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        
        episode_reward += reward
        step_num += 1

        if render:
            env.render()

        print(f"steps: {step_num} reward: {episode_reward} done: {done} info: {info}")
        
    return step_num, episode_reward, done, info
