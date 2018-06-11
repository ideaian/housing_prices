import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ConvertDates(BaseEstimator, TransformerMixin):
    #: TODO: Generalize this to allow for different date-type outputs to become features
    def __init__(self, date_fields):
        self.date_fields = date_fields
        self.new_date_fields = []

    def fit(self, df, y):
        return self

    def transform(self, df):
        df, new_date_fields = convert_dates_helper(df, self.date_fields)
        self.new_date_fields = new_date_fields
        return df


def convert_dates_helper(df, date_fields):
    # This function will convert date strings into numerical
    # forms that are more amenable for use by ML models
    if date_fields is None:
        return df

    new_fields = []
    for field in date_fields:
        if field not in df.keys():
            print("Warning, field {} not in dataframe".format(field))
            continue
        d = pd.Series(pd.to_datetime(df[field], format='%Y-%m-%d'))

        df[field + 'DayOfWeek'] = d.dt.dayofweek
        df[field + 'WeekOfYear'] = d.dt.weekofyear
        df[field + 'Month'] = d.dt.month
        df[field + 'Year'] = d.dt.year
        new_fields.append(field + 'DayOfWeek')
        new_fields.append(field + 'WeekOfYear')
        new_fields.append(field + 'Month')
        new_fields.append(field + 'Year')

    return df, new_fields


class ZeroToNanFields(BaseEstimator, TransformerMixin):
    def __init__(self, zero_to_nan_fields):
        self.zero_to_nan_fields = zero_to_nan_fields

    def fit(self, df, y):
        return self

    def transform(self, df):
        # Replace zeros with nan for imputation down the way
        if self.zero_to_nan_fields is not None:
            for field in self.zero_to_nan_fields:
                if field in df.keys():
                    df[df[field] == 0, field] = np.nan
        return df


class NanToZeroFields(BaseEstimator, TransformerMixin):
    def __init__(self, nan_to_zero_fields):
        self.nan_to_zero_fields = nan_to_zero_fields

    def fit(self, df, y):
        return self

    def transform(self, df):
        # Replace zeros with nan for imputation down the way
        if self.nan_to_zero_fields is not None:
            for field in self.nan_to_zero_fields:
                if field in df.keys():
                    df.loc[pd.isnull(df[field]), field] = 0
        return df


class KeepRequiredFields(BaseEstimator, TransformerMixin):
    def __init__(self, required_fields):
        self.required_fields = required_fields

    def fit(self, df, y):
        return self

    def transform(self, df):
        for key in iter(df.keys()):
            if key not in self.required_fields:
                df.drop(labels=key, axis=1, inplace=True)
        return df


class RemoveNotUsefulFields(BaseEstimator, TransformerMixin):
    def __init__(self, not_useful_fields):
        self.not_useful_fields = not_useful_fields

    def fit(self, df, y):
        return self

    def transform(self, df):
        return remove_not_useful_fields(df, self.not_useful_fields)


def remove_not_useful_fields(df, not_useful_fields=None):
    # This function will remove entire columns of a dataframe that are
    # not needed for the analysis (meta or highly duplicated info)
    if not_useful_fields is None:
        return df

    try:  # Handle different versions of pandas
        df.drop(columns=not_useful_fields, inplace=True)
    except TypeError:
        df.drop(not_useful_fields, axis=1, inplace=True)

    return df


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y):
        return self
    def transform(self, X):
        return X


class Dropout(BaseEstimator, TransformerMixin):
    '''
    Randomly null values based on dropout_rate
    '''
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
    def fit(self, X, y):
        return self
    def transform(self, X):
        mask = np.reshape(np.random.random_sample(size=np.prod(X.shape))>self.dropout_rate, X.shape)
        X[mask] = 0
        return X
