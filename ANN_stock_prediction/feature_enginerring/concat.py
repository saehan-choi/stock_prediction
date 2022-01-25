import pandas as pd
from torch import concat

coin = ['ATOM', 'BTC', 'ETH', 'GXC', 'HIBS', 'KLAY', 'LUNA', 'XRP']

path = './input/'

df1 = pd.read_csv(path+'ATOM_train.csv')
df2 = pd.read_csv(path+'BTC_train.csv')
df3 = pd.read_csv(path+'ETH_train.csv')
df4 = pd.read_csv(path+'GXC_train.csv')
df5 = pd.read_csv(path+'HIBS_train.csv')
df6 = pd.read_csv(path+'KLAY_train.csv')
df7 = pd.read_csv(path+'LUNA_train.csv')
df8 = pd.read_csv(path+'XRP_train.csv')



df9 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],ignore_index=True)

# print(df9)

df9.to_csv(path+'train.csv', index=False)

df1 = pd.read_csv(path+'ATOM_validation.csv')
df2 = pd.read_csv(path+'BTC_validation.csv')
df3 = pd.read_csv(path+'ETH_validation.csv')
df4 = pd.read_csv(path+'GXC_validation.csv')
df5 = pd.read_csv(path+'HIBS_validation.csv')
df6 = pd.read_csv(path+'KLAY_validation.csv')
df7 = pd.read_csv(path+'LUNA_validation.csv')
df8 = pd.read_csv(path+'XRP_validation.csv')

df9 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],ignore_index=True)

df9.to_csv(path+'validation.csv', index=False)

df1 = pd.read_csv(path+'ATOM_test.csv')
df2 = pd.read_csv(path+'BTC_test.csv')
df3 = pd.read_csv(path+'ETH_test.csv')
df4 = pd.read_csv(path+'GXC_test.csv')
df5 = pd.read_csv(path+'HIBS_test.csv')
df6 = pd.read_csv(path+'KLAY_test.csv')
df7 = pd.read_csv(path+'LUNA_test.csv')
df8 = pd.read_csv(path+'XRP_test.csv')

df9 = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8],ignore_index=True)

df9.to_csv(path+'test.csv', index=False)
