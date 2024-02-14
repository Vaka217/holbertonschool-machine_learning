#!/usr/bin/env python3
"""Q Learning Module"""
import numpy as np


def play(env, Q, max_steps=100):
    """The trained agent play an episode:

        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        max_steps is the maximum number of steps in the episode
        Each state of the board should be displayed via the console
        You should always exploit the Q-table
        Returns: the total rewards for the episode"""
    state = env.reset()[0]
    episode_reward = 0

    for step in range(max_steps):
        env.render()

        action = np.argmax(Q[state, :])

        new_state, reward, terminated, truncated, info = env.step(action)

        if terminated and reward == 0:
            reward = -1

        state = new_state
        episode_reward += reward

        if terminated:
            env.render()
            return episode_reward

    env.close()
