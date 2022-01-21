import time
import pybithumb
from collections import deque
from datetime import datetime
import pandas as pd

times = deque(maxlen=10)
times.append(0)

err_ = []
price_ = []
time_ = []



for i in range(10000000):
    tm = time.localtime(time.time())
    times.append(tm.tm_sec)    
    
    if times[1]-times[0]==1 or times[1]-times[0]==-59:
        price = pybithumb.get_current_price("BTC")
        print(price)
        price_.append(price)
        time_.append(datetime.now())
        err_.append('right')
    
    elif times[1]-times[0] > 1 or -1 >= times[1]-times[0] > -59:
        price = pybithumb.get_current_price("BTC")
        price_.append(price)
        time_.append(datetime.now())
        err_.append('err')


df1 = pd.DataFrame({'err': err_, \
        'price': price_, \
        'time': time_})

df1.to_csv("filename.csv", mode='w')

