#!/usr/bin/env python3
"""Evaluates NN"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_path + ".meta")
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        y_pred = tf.get_collection("tensors")[0]
        loss = tf.get_collection("tensors")[1]
        accuracy = tf.get_collection("tensors")[2]
        x = tf.get_collection("placeholders")[0]
        y = tf.get_collection("placeholders")[1]
        y_pred, accuracy, loss = sess.run([y_pred, accuracy, loss],
                                          feed_dict={x: X, y: Y})
    return y_pred, accuracy, loss
