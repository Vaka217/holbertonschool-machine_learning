#!/usr/bin/env python3
"""Monte Carlo Module"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm:

    env is the openAI environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next
    action to take
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate

    Returns: V, the updated value estimate"""
    for episode in range(episodes):
        states = []
        rewards = []
        state = env.reset()[0]

        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            states.append(state)
            rewards.append(reward)

            if terminated:
                break
            state = new_state

        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            V[states[t]] = V[states[t]] + alpha * (G - V[states[t]])

    return V
