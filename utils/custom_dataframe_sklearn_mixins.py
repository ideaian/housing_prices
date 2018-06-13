import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MakeFractionalFeatures(BaseEstimator, TransformerMixin):
    # can also return the field names
    def __init__(self, fractional_feature_list, remove_paired_features=False):
        #: A fractional feature list tuple where you divide the first by the second
        self.fractional_feature_list = fractional_feature_list
        self.removed_paired_features = remove_paired_features
        self.new_feature_list = [] 
    def fit(self, df, y):
        return self
    def transform(self, df):
        #: Check to ensure feature lists is in the data frame

        #: go through the feature list and make a new dataframe element 
        for i, (field1, field2) in enumerate(self.fractional_feature_list):
            new_field_name = field1 + '_per_' + field2
            df[new_field_name] = df[field1] / df[field2]
            self.new_feature_list.append(new_field_name)

        if self.remove_paired_features:
            df.drop(labels=key, axis=1, inplace=True)

        return df


class ConvertDates(BaseEstimator, TransformerMixin):
    # This Transformer converts datestrings to numerical values
    # corresponding to different time periods.
    #: TODO: Generalize this to allow for different type of dates to be output
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
        #TODO: Detect Datestring Formats
        d = pd.Series(pd.to_datetime(df[field], format='%Y-%m-%d'))
        
        #TODO: Make this general for other types
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
        for key in self.required_fields:
            if key not in df.keys():
                import sys
                sys.exit("Required field {} not found in df list {}".format(key, df.keys()))

        #for key in iter(df.keys()):
        #    if key not in self.required_fields:
        #        print("Dropping field {}".format(key))
        #        df.drop(labels=key, axis=1, inplace=True)
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
