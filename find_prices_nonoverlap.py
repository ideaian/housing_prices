import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import( 
    LabelEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, LabelEncoder, Imputer
)
# CategoricalEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#from sklearn.ensemble import LGBMRegressor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
            
from sklearn.base import BaseEstimator, TransformerMixin
        
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(len(df))
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    df_train = df.iloc[perm[:train_end]]
    df_validate = df.iloc[perm[train_end:validate_end]]
    df_test = df.iloc[perm[validate_end:]]
    return df_train, df_validate, df_test
    

def prepare_data(df, not_useful_fields=None,  
                date_fields=None, 
                remove_outliers=False, 
                required_fields=None,
                zero_to_nan_fields=None,
                nan_to_zero_fields=None, 
                categorical_fields=None):

    # This function will clean the data
       
    # All, remove duplicates
    df.drop_duplicates(inplace=True)

    # Remove entire fields as they are not needed. 
    df = remove_not_useful_fields(df, not_useful_fields)

    # Remove rows with fields as nan
    # This will drop rows if the specified field is Nan. 
    if required_fields is not None:
        #import ipdb; ipdb.set_trace()
        for field in required_fields:
            df.drop(df.index[pd.isnull(df[field])], inplace=True)

    # Replace zeros with nan for imputation down the way
    if zero_to_nan_fields is not None:
        for field in zero_to_nan_fields:
            df[field][df[field] == 0] = np.nan
    # Replace nans with zeros for pseudo classification (due to nan's being present)
    if nan_to_zero_fields is not None:
        for field in nan_to_zero_fields:
            print("Nulling field {}".format(field))

            df.loc[pd.isnull(df[field]),'field'] = 0
            #df[field][pd.isnull(df[field])] = 0
    # Remove outlier values
    if remove_outliers:
        print('Not Implemented')
        None

    df, new_fields = convert_dates(df, date_fields)
    df, new_categorical_fields = convert_to_categorical_features(df, categorical_fields)

    return df, new_fields+new_categorical_fields


class TransformZipCodes(BaseEstimator, TransformerMixin):
    # This class will make zipcodes become 
    # for data not containing the zipcode, it will find the 
    # center of zipcode lat-long data and estimate a zipcode

    # A better way would be to use a web API to search the address and extract the zipcode
    # I'm going to do this first.

    # For the future: make this a general imputer that provides classification for training data
    # using clustering methods. 
    
    def __init__(self):
        self.zip_code_lat_long_df = pd.DataFrame()

    def fit(self, X):
        self.zip_code_lat_long_df = \
            pd.groupby(X,'zipcode')['latitude','longitude'].mean().add_suffix('_mean') 
        return self

    def transform(self, X):
        bad_index = ~X['zipcode'].isin(self.zip_code_lat_long_df.index)
        dd = X[bad_index][['latitude','longitude']]

        if len(dd) == 0:
            # No unseen zipcodes were found
            return X
        
        ndx =(dd[['latitude','longitude']]).reset_index().apply(
                                    lambda x: smallest_gcd_distance_ndx(
                                        self.zip_code_lat_long_df.latitude_mean.values, 
                                        self.zip_code_lat_long_df.longitude_mean.values, 
                                        x[0],x[1]), 
                                    axis=1)
        X.loc[bad_index,'zipcode'] = \
            self.zip_code_lat_long_df.iloc[ndx].index
        return X


def smallest_gcd_distance_ndx(lat1, lng1, lat2, lng2):
    '''
    find the array index of the smallest great-circal distance
    '''
    return np.argmin(gcd_vec(lat1, lng1, lat2, lng2))
    
def gcd_vec(lat1, lng1, lat2, lng2):
    '''
    Calculate great circle distance.
    http://www.johndcook.com/blog/python_longitude_latitude/

    Parameters
    ----------
    lat1, lng1, lat2, lng2: float or array of float

    Returns
    -------
    distance:
      distance from ``(lat1, lng1)`` to ``(lat2, lng2)`` in kilometers.
    '''
    # python2 users will have to use ascii identifiers
    f1 = np.deg2rad(90 - lat1)
    f2 = np.deg2rad(90 - lat2)

    t1 = np.deg2rad(lng1)
    t2 = np.deg2rad(lng2)

    cos = (np.sin(f1) * np.sin(f2) * np.cos(t1 - t2) +
           np.cos(f1) * np.cos(f2))
    arc = np.arccos(cos)
    return arc * 6373


def convert_to_categorical_features(df, categorical_fields):
    # There is possibly a better way of doing this, but with dataframe_mapper and sklearn 
    # categorical features doesn't work. To revise this involves changing the code to
    # not rely on dataframe_mapper for featurize

    return df, []


def remove_not_useful_fields(df, not_useful_fields=None):
    # This function will remove entire columns of a dataframe that are 
    # not needed for the analysis (meta or highly duplicated info)
    if not_useful_fields is None:
        return df

    try: #Handle different versions of pandas
        df.drop(columns=not_useful_fields, inplace=True)
    except TypeError:
        df.drop(not_useful_fields, axis=1, inplace=True)

    return df


def convert_dates(df, date_fields):
    # This function will convert date strings into numerical
    # forms that are more amenable for use by ML models
    new_fields=[]
    for field in date_fields:
        d = pd.Series(pd.to_datetime(df[field],format='%Y-%m-%d'))
        
        df[field+'DayOfWeek'] = d.dt.dayofweek
        df[field+'WeekOfYear'] = d.dt.weekofyear
        df[field+'Month'] = d.dt.month
        df[field+'Year'] = d.dt.year
        new_fields.append(field+'DayOfWeek')
        new_fields.append(field+'WeekOfYear')
        new_fields.append(field+'Month')
        new_fields.append(field+'Year')
    
    return remove_not_useful_fields(df, date_fields), new_fields
        

def featurize(features):
    
    day_range = [0,6]
    week_range = [0,52]
    month_range = [0,11]
    year_range = [1900, 2020]
    transformations = [
        ('zipcode',LabelEncoder()),
        ('latitude', StandardScaler()),
        ('logitude', StandardScaler()),
        ('bedrooms', StandardScaler()),
        ('bathrooms', StandardScaler()),
        ('rooms', StandardScaler()),
        ('squareFootage', StandardScaler()),                                       
        ('lotSize', StandardScaler()),
        ('lastSaleAmount', StandardScaler()),
        ('lastSaleDayOfWeek', MinMaxScaler()),
        ('lastSaleWeekOfYear',MinMaxScaler()),
        ('lastSaleMonth',MinMaxScaler()),
        ('lastSaleYear',MinMaxScaler()),
        ('priorSaleAmount', StandardScaler()),
        ('priorSaleDayOfWeek',MinMaxScaler()),
        ('priorSaleWeekOfYear',MinMaxScaler()),
        ('priorSaleMonth',MinMaxScaler()),
        ('priorSaleYear',MinMaxScaler()),
    ]
    return DataFrameMapper(filter(lambda x: x[0] in features, transformations))


def print_metrics(y, y_pred):
    print("Sqrt mse: {}".format(np.sqrt(metrics.mean_squared_error(y,y_pred))))
    print("Mean absolute error: {}".format(metrics.mean_absolute_error(y,y_pred)))
    print("R2 score: {}".format(metrics.r2_score(y,y_pred)))
    print("Absolute mean relative error: {}".format(abs_mean_relative_error(y,y_pred))


def abs_mean_relative_error(y,y_pred):
        return np.mean(np.abs(y-y_pred)/(y))

#TODO: Zipcode onehot encoding or with LabelEncoding
#TODO: Evaluate different feature sets. 
#TODO: error handling of input data types
#TODO: Ensure more broad testing of input data types (Nan/Zero handling)
#TODO: Improve plotting/metric evaluation
#TODO: Github repository
#TODO: Robust deployment on different machine

# IDEA: Cluster based on missing data!
# Ensure zipcode
