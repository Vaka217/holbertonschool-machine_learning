#!/usr/bin/env python3
"""Trains a Convolutional Neural Network to classify the CIFAR 10 dataset"""

import tensorflow.keras as K

def preprocess_data(X, Y):
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p

(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
(x_train, y_train) = preprocess_data(x_train, y_train)
(x_test, y_test) = preprocess_data(x_test, y_test)

lambtha = K.layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, data_format='channels_last', interpolation='bilinear'))

base_model = K.applications.ResNet50(include_top=False, weights='imagenet')
base_model.trainable = False

model = K.Sequential()
model.add(lambtha)
model.add(base_model)

model.add(K.layers.Flatten())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(256, activation='relu', kernel_initializer=K.initializers.HeNormal))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu', kernel_initializer=K.initializers.HeNormal))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.HeNormal))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(10, activation='softmax', kernel_initializer=K.initializers.HeNormal))

model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics=['accuracy'])
checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5', 'val_accuracy', verbose=1, save_best_only=True)
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), callbacks=[checkpoint])
model.save('cifar10.h5')
