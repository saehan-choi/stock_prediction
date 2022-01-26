import pandas as pd


test = pd.read_csv('SAND_new.csv')


print(test[['high']].max()-test[['high']].min())

# 