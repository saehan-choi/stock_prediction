import pybithumb
import pandas as pd

arr = ['BTC', 'ETH', 'XRP', 'ATOM', 'LUNA', 'HIBS', 'KLAY', 'GXC']

for i in arr:

    df = pybithumb.get_candlestick(i, chart_intervals="5m")
    df.to_csv(f"{i}.csv", mode='w')

