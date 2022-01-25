import pandas as pd
import pybithumb

count = 3000

coins = ['SAND']

for coin in coins:

    df = pybithumb.get_candlestick(coin, chart_intervals="5m")
    df = df.tail(count)
    df.to_csv(f"{coin}_new.csv", mode='w')
    df = pd.read_csv(f"{coin}_new.csv")
    
    df[['trash1', 'trash2', 'time']] = df['time'].str.partition(sep=" ")
    df[['hour', 'minute', 'second']] = df['time'].str.split(':', n=3, expand=True)
    df['hour'] = pd.to_numeric(df['hour'])
    df['minute'] = pd.to_numeric(df['minute'])

    df['time'] = ((df['hour'] * 12) + (df['minute'] / 5) + 1) / 288
    df = df.drop(['hour', 'minute', 'second', 'trash1', 'trash2'], axis='columns')

    for n in range(1, 20):
        for i in df.index:
            if i < count - 19:
                df.loc[i, f'open{n}'] = df.loc[i + n, 'open']
                df.loc[i, f'high{n}'] = df.loc[i + n, 'high']
                df.loc[i, f'low{n}'] = df.loc[i + n, 'low']
                df.loc[i, f'close{n}'] = df.loc[i + n, 'close']
                df.loc[i, f'volume{n}'] = df.loc[i + n, 'volume']
                
    df['close'] = pd.to_numeric(df['close'])

    for i in df.index:
        if i < count - 20:
            df.loc[i, 'rate'] = ((df.loc[i + 20, 'close'] / df.loc[i + 19, 'close']) - 1) * 100
            
        
        if i < count - 22:
            df.loc[i, 'rate_after'] = ((df.loc[i + 22, 'close'] / df.loc[i + 19, 'close']) - 1) * 100

        # 'rate_after' 말고 'rate'로 고치면 5분뒤 데이터 예측임
        if 0 < df.loc[i, 'rate_after'] <= 0.25:
            df.loc[i, 'g_up'] = 0
            df.loc[i, 'up'] = 0
            df.loc[i, 'l_up'] = 1
            df.loc[i, 'l_down'] = 0
            df.loc[i, 'down'] = 0
            df.loc[i, 'g_down'] = 0
        
        elif 0.25 < df.loc[i, 'rate_after'] <= 0.5:
            df.loc[i, 'g_up'] = 0
            df.loc[i, 'up'] = 1
            df.loc[i, 'l_up'] = 0
            df.loc[i, 'l_down'] = 0
            df.loc[i, 'down'] = 0
            df.loc[i, 'g_down'] = 0
        
        elif 0.5 < df.loc[i, 'rate_after']:
            df.loc[i, 'g_up'] = 1
            df.loc[i, 'up'] = 0
            df.loc[i, 'l_up'] = 0
            df.loc[i, 'l_down'] = 0
            df.loc[i, 'down'] = 0
            df.loc[i, 'g_down'] = 0

        elif -0.25 < df.loc[i, 'rate_after'] <= 0:
            df.loc[i, 'g_up'] = 0
            df.loc[i, 'up'] = 0
            df.loc[i, 'l_up'] = 0
            df.loc[i, 'l_down'] = 1
            df.loc[i, 'down'] = 0
            df.loc[i, 'g_down'] = 0

        elif -0.5 < df.loc[i, 'rate_after'] <= -0.25:
            df.loc[i, 'g_up'] = 0
            df.loc[i, 'up'] = 0
            df.loc[i, 'l_up'] = 0
            df.loc[i, 'l_down'] = 0
            df.loc[i, 'down'] = 1
            df.loc[i, 'g_down'] = 0

        elif df.loc[i, 'rate_after'] <= -0.5:
            df.loc[i, 'g_up'] = 0
            df.loc[i, 'up'] = 0
            df.loc[i, 'l_up'] = 0
            df.loc[i, 'l_down'] = 0
            df.loc[i, 'down'] = 0
            df.loc[i, 'g_down'] = 1

    df = df.dropna()
    df.to_csv(f'{coin}_new.csv', index=False)