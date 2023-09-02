#!/usr/bin/env python3
"""Builds, Trains, and Saves NN using mini-batch gradient descent"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded NN model using mini-batch gradient descent"""

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(load_path + ".meta")
        new_saver.restore(sess, load_path)

        train_op = tf.get_collection("train_op")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        m = X_train.shape[0]
        batches = m // batch_size

        for epoch in range(epochs + 1):
            print(f"After {epoch} epochs:")

            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={
                x: X_train, y: Y_train})

            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")

            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={
                x: X_valid, y: Y_valid})

            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

            if epoch < epochs:
                X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

                for step in range(batches):
                    start = step * batch_size
                    end = start + batch_size if start + batch_size < m else m

                    Y_batch = Y_shuffle[start:end]
                    X_batch = X_shuffle[start:end]
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    if step != 0 and (step + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                        print(f"\tStep {step + 1}:")
                        print(f"\t\tCost: {step_cost}")
                        print(f"\t\tAccuracy: {step_accuracy}")

        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)
        return save_path
