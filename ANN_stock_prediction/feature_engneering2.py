import pandas as pd

# file load
df = pd.read_csv('filename.csv')
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# 필요없는열은 제거

def remove_error_value(df, error_value):

    idx_err = df[df[error_value]==error_value].index
    df = df.drop(idx_err)
    # print(df)
    return df


def sort_drop(df):
    df = df.reset_index(drop=True)
    df = df.drop([df.index[0], df.index[1]])
    # 0번째와 1번째값들이 정확한 1초간격이 아닌경우가 있어서 이거 해줬습니다
    df = df.reset_index(drop=True)
    print(df)
    return df


df = remove_error_value(df, 'err')
df = sort_drop(df)


# print(df)

