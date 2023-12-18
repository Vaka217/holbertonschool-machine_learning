#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd


def preprocess_data(csv_path):

    df = pd.read_csv(csv_path)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.sort_values(by=['Timestamp'], inplace=True, ascending=True)

    df = df.drop(['High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)'], axis=1)
    df = df.dropna()

    df = df[df["Timestamp"] >= "2017"]
    df.set_index('Timestamp', inplace=True)

    df = df.resample('H').mean()

    print(df.head())

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    train_df = train_df.diff()
    val_df = val_df.diff()
    test_df = test_df.diff()

    return train_df, val_df, test_df
