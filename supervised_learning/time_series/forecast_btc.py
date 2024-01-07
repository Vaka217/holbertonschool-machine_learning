#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from preprocess_data import preprocess_data


def create_dataset(df):
    data = df.values
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=sequence_length + 1,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size
    )
    return dataset

csv_path = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
train_df, val_df, test_df = preprocess_data(csv_path)

sequence_length = 24  # Number of hours to look back for prediction
batch_size = 64

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)
test_dataset = create_dataset(test_df)

model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(sequence_length, train_df.shape[1]), return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, validation_data=val_dataset, epochs=10)

mse = model.evaluate(test_dataset)
print(f"Test MSE: {mse}")
