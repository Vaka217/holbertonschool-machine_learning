#!/usr/bin/env python3
"""TD Lambda Module"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(λ) algorithm:

    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next
    action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate

    Returns: V, the updated value estimate"""
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            delta = reward + gamma * V[new_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility

            eligibility *= lambtha * gamma

            if terminated:
                break

            state = new_state

    return V
