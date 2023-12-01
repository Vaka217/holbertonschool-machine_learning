#!/usr/bin/env python3
import GPyOpt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

iris = load_iris()
X, y = iris.data, iris.target

label_encoder = LabelEncoder()
y_binary = label_encoder.fit_transform(y)
y_binary = (y_binary == 0).astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, random_state=42)


def objective_function(params):
    learning_rate, num_units, dropout_rate, l2_reg, batch_size = params[0]

    batch_size = int(batch_size)

    model = Sequential()
    model.add(Dense(num_units, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid',
              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))

    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_path = f"checkpoint_lr{learning_rate}_nu{num_units}_dr{dropout_rate}_l2{l2_reg}_bs{batch_size}.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=5, mode='max')

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=batch_size, callbacks=[checkpoint, early_stopping])

    return max(history.history['val_accuracy'])


bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'l2_reg', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=objective_function, domain=bounds)

optimizer.run_optimization(max_iter=30, report_file='bayes_opt.txt')

optimizer.plot_convergence()
