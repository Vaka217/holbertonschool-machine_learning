#!/usr/bin/env python3
"""Builds, trains, and saves a neural network model"""
import tensorflow.compat.v1 as tf
import numpy as np


def forward_prop(prev, layers, activations, epsilon):
    """Forward Propagation with batch normalization"""
    # all layers get batch_normalization but the last one, that stays without
    # any activation or normalization
    for i in range(len(layers)):
        heetal = tf.keras.initializers.VarianceScaling(mode='fan_avg')

        layer = tf.keras.layers.Dense(layers[i], activation=activations[i],
                                      kernel_initializer=heetal)
        Z = layer(prev)

        if i == len(layers) - 1:
            prev = Z
            return prev

        beta = tf.Variable(initial_value=tf.zeros(shape=[layers[i]]),
                           name="beta")
        gamma = tf.Variable(initial_value=tf.ones(shape=layers[i]),
                            name="gamma")

        m, v = tf.nn.moments(Z, axes=[0])

        prev = tf.nn.batch_normalization(Z, mean=m, variance=v, offset=beta,
                                         scale=gamma, variance_epsilon=epsilon)

        if activations is not None:
            prev = activations(prev)


def shuffle_data(X, Y):
    """Shuffles X and Y"""
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay, and
    batch normalization"""
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float32, shape=[None, len(X_train[1])], name="x")
    y = tf.placeholder(tf.float32, shape=[None, len(Y_train[1])], name="y")
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection("y_pred", y_pred)

    # intialize loss and add it to collection
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection("loss", loss)

    # intialize accuracy and add it to collection
    pred_labels = tf.argmax(y_pred, axis=1)
    correct_pred = tf.equal(pred_labels, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.add_to_collection("accuracy", accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    decay_step = decay_rate * 10

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    train_op = optimizer.minimize(loss, global_step)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        m = X_train.shape[0]

        for epoch in range(epochs):
            # print training and validation cost and accuracy
            print(f"After {epoch} epochs:")

            train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={
                x: X_train, y: Y_train})

            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")

            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={
                x: X_valid, y: Y_valid})

            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

            # shuffle data
            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

            for step in range(0, m, batch_size):
                # get X_batch and Y_batch from X_train shuffled
                # and Y_train shuffled
                start = step * batch_size
                end = start + batch_size if start + batch_size < m else m

                X_batch = X_shuffle[start:end]
                Y_batch = Y_shuffle[start:end]

                # run training operation
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # print batch cost and accuracy
                if step != 0 and step % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

        # print training and validation cost and accuracy again
        print(f"After {epoch} epochs:")

        train_cost, train_accuracy = sess.run([loss, accuracy], feed_dict={
            x: X_train, y: Y_train})

        print(f"\tTraining Cost: {train_cost}")
        print(f"\tTraining Accuracy: {train_accuracy}")

        valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_dict={
            x: X_valid, y: Y_valid})

        print(f"\tValidation Cost: {valid_cost}")
        print(f"\tValidation Accuracy: {valid_accuracy}")

        # save and return the path to where the model was saved
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)
        return save_path
