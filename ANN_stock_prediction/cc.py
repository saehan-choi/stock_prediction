import pandas as pd
from transformers import data

# from pandas import DataFrame

import pandas as pd


df1 = pd.DataFrame({'Color': ['blue', 'green', 'red', 'black'], \
        'Product ID': [1, 2, 3, 4], \
        'Product Name': ['t-shirt', 't-shirt', 'skirt', 'skirt']})

df1.to_csv("filename.csv", mode='w')

