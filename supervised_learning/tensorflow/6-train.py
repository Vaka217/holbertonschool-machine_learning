#!/usr/bin/env python3
"""Builds, Trains, and Saves NN"""


import tensorflow as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
	"""Builds, trains, and saves a neural network classifier"""
	x, y = create_placeholders(X, Y)

	y_pred = forward_prop(x_train, layer_sizes, activations)

	loss_train = calculate_loss(y, y_pred)

	train_op = create_train_op(loss, alpha)

	accuracy = calculate_accuracy(y, y_pred)

	tf.add_to_collection("placeholders", x)
	tf.add_to_collection("placeholders", y)
	tf.add_to_collection("tensors", y_pred)
	tf.add_to_collection("tensors", loss)
	tf.add_to_collection("tensors", accuracy)
	tf.add_to_collection("operation", train_op)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(iterations):
			train_loss, train_accuracy = sess.run([loss, accuracy], feed_dict={X: X_train, Y: Y_train})
			valid_loss, valid_accuracy = sess.run([loss, accuracy], feed_dict={X: X_valid, Y: Y_valid})
			if i % 100 == 0:
				print("After {} iterations:\n\tTraining Cost: {}\n\tTraining Accuracy: {}\n\tValidation Cost: {}\n\tValidation Accuracy: {}".format(i, train_loss, train_accuracy, valid_loss, valid_accuracy))
		saver = tf.train.Saver()
		saver.save(sess, save_path)

