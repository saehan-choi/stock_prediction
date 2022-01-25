import FinanceDataReader as fdr

df_NASDAQ = fdr.StockListing('NASDAQ')

# print(df_NASDAQ.head())


print(df_NASDAQ)
print(df_NASDAQ.shape)


df_NASDAQ.to_csv("NASDAQ.csv",index=False)