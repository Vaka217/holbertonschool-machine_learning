#!/usr/bin/env python3
"""Determines if you should stop gradient descent early"""
import tensorflow.compat.v1 as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early
    Early stopping should occur when the validation cost of the network has
    not decreased relative to the optimal validation cost by more than the
    threshold over a specific patience count

    Arguments:
        cost -- current validation cost
        opt_cost -- optimal validation cost
        threshold -- the allowed max difference between cost and opt_cost
        patience -- patience count for early stopping
        count -- count of how long the threshold has not been met

    Returns:
        bool -- Wheter the network should be stopped early
        count -- updated count
    """
    count = 0 if opt_cost - cost > threshold else count + 1
    if count != patience:
        return False, count
    return True, count
