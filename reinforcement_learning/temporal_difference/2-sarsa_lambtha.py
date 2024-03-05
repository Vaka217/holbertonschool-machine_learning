#!/usr/bin/env python3
"""SARSA Module"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs SARSA(λ):

    env is the openAI environment instance
    Q is a numpy.ndarray of shape (s,a) containing the Q table
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes

    Returns: Q, the updated Q table"""
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(Q)
        action = epsilon_greedy(Q, state, epsilon)

        for _ in range(max_steps):
            new_state, reward, terminated, truncated, info = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility

            eligibility *= lambtha * gamma

            if terminated:
                break

            state = new_state
            action = new_action

        epsilon = max(epsilon - epsilon_decay, min_epsilon)

    return Q

def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action:

        Q is a numpy.ndarray containing the q-table
        state is the current state
        epsilon is the epsilon to use for the calculation
        You should sample p with numpy.random.uniformn to determine if your
        algorithm should explore or exploit
        If exploring, you should pick the next action with numpy.random.randint
        from all possible actions

        Returns: the next action index"""

    explore_threshold = np.random.uniform(0, 1)

    if explore_threshold > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])

    return action
