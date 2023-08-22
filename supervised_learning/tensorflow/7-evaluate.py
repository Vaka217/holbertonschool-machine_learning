#!/usr/bin/env python3
"""Evaluates NN"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph("./{}".format(save_path))
        new_saver.restore(sess, "./{}".format(save_path))
        y_pred = tf.get_collection("y_pred")
        loss = tf.get_collection("loss")
        accuracy = tf.get_collection("accuracy")
        x = tf.get_collection("x")
        y = tf.get_collection("y")
        y_pred, accuracy, loss = sess.run([y_pred, accuracy, loss],
                                          feed_dict={x: X, y: Y})
    return y_pred, accuracy, loss
