#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.iloc[:, :-1]
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")
df["Close"].ffill(inplace=True)
df = df.fillna(value={"Volume_(BTC)": 0,
                      "Volume_(Currency)": 0,
                      "Low": df["Close"],
                      "Open": df["Close"],
                      "High": df["Close"]})

df = df['2017':].resample("D").agg({'High': 'max', 'Low': 'min', 'Open': 'mean', 'Close': 'mean', 'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum'})
plot = df.plot()
fig = plot.get_figure()
fig.savefig("aaa.png")
