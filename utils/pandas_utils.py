import numpy as np


def train_validate_test_split(df, train_percent=.8, validate_percent=.05, seed=None):

    np.random.seed(seed)
    perm = np.random.permutation(len(df))
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    df_train = df.iloc[perm[:train_end]]
    df_validate = df.iloc[perm[train_end:validate_end]]
    df_test = df.iloc[perm[validate_end:]]
    return df_train, df_validate, df_test

