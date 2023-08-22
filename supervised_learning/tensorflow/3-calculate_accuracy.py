#!/usr/bin/env python3
"""Accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    pred_labels = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(pred_labels, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy
