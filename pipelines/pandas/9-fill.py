#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.iloc[:, :-1]
df["Close"].ffill(inplace=True)
df = df.fillna(value={"Volume_(BTC)": 0,
                      "Volume_(Currency)": 0,
                      "Low": df["Close"],
                      "Open": df["Close"],
                      "High": df["Close"]})
print(df.head())
print(df.tail())
