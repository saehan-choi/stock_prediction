import pandas as pd

err_ = ['right', 'right', 'right', 'right', 'right', 'right','right']
price_ = [6423530, 6423530, 6464530, 6423630, 6423580, 6423564, 6423565]
time_ = [56, 57, 58, 59, 60, 61, 86400]

df = pd.DataFrame({'err': err_, \
        'price': price_, \
        'time': time_})

df.to_csv("holy_test.csv", mode='w')


df = pd.read_csv('holy_test.csv', index_col=[0])
# unnamed 없앰
# print(df)

df_1 = pd.read_csv('holy_test.csv', index_col=[0]).copy()

for i in range(len(df)):
        df_1['time'][i] = str(format(df['time'][i], 'b'))
        

        # print(df['time'][i])

print(df_1)


