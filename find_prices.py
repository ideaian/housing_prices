import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import( 
    LabelEncoder, LabelBinarizer, MinMaxScaler, StandardScaler, OneHotEncoder, Imputer
)
# CategoricalEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestRegressor


def prepare_test_train_validation(df, split_by_zipcode=True, seed=1234, 
                                  train_percent=0.6, validate_percent=0.2):
    min_needed_by_zipcode = 10
    df_train, df_test, df_validate = pd.DataFrame()
    if split_by_zipcode:
        zipcodes = df['zipcode'].unique()
        for zipcode in zipcodes:
            df_zip = df[df['zipcode']==zipcode]
            out = train_validate_test_split(df_zip, 
                                            train_percent=train_percent, 
                                            validate_percent=validate_percent, 
                                            seed=seed)
            df_train.append(out[0], ignore_index=True)
            df_test.append(out[1], ignore_index=True)
            df_validate.append(out[2], ignore_index=True)
    else:
        df_train, df_test, df_validate = \
            train_validate_test_split(df, 
                                      train_percent=train_percent, 
                                      validate_percent=validate_percent, 
                                      seed=seed)
    return df_train, df_test, df_validate
        
            
        
def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    df_train = df.iloc[perm[:train_end]]
    df_validate = df.iloc[perm[train_end:validate_end]]
    df_test = df.iloc[perm[validate_end:]]
    return df_train, df_validate, df_test
        
        
def remove_fields(df, drop_for_prediction=False):
#     #All, remove duplicates
#     df.drop_duplicates(inplace=True)
    
#     #Check for out of range values
    
#     #Needs to be in Denver, CO, 
#     # ranges cannot be negative, or extremely high
    
    
#     #TODO: 

#     if drop_for_prediction:
#         df.drop[columns='id']
#         #Address:
#         # do some embedding or get major street, likely duplicated information as latlong
#         # DROP for NOW
#         df.drop(columns=['address'])

#         #City/State:
#         # remove, all the same. 
#         df.drop(columns=['city','state'])
    bad_columns = ['id','city','state', 'address']
    try:
        df = df.drop(columns=bad_columns)
    except TypeError:
        df = df.drop(bad_columns, axis=1)
    return df

def convert_dates(df):
    d = pd.Series(pd.to_datetime(df['lastSaleDate'],format='%Y-%m-%d'))
    df['lastSaleDayOfWeek'] = d.dt.dayofweek
    df['lastSaleWeekOfYear'] = d.dt.weekofyear
    df['lastSaleMonth'] = d.dt.month
    df['lastSaleYear'] = d.dt.year
    d = pd.Series(pd.to_datetime(df['priorSaleDate'],format='%Y-%m-%d'))
    df['priorSaleDayOfWeek'] = d.dt.dayofweek
    df['priorSaleWeekOfYear'] = d.dt.weekofyear
    df['priorSaleMonth'] = d.dt.month
    df['priorSaleYear'] = d.dt.year
    try:
        df = df.drop(columns=['lastSaleDate','priorSaleDate'])
    except TypeError:
        df = df.drop(['lastSaleDate','priorSaleDate'], axis=1)
    return remove_fields(df)
        
        
def featurize(features):
    #Zipcode: DO NOT NORMALIZE
    # Warning if new zipcode is not present

    #Lat/Long: normalize
    #Bedrooms/Bathrooms/Rooms: normalize or not
    #Square Footage: normalize
    #lotSize: Normalize
    #yearBuilt: nothing
    #lastSaleAmount: Replace zero with nan
    #lastSaleDate: Split
    # lastSaleDayOfWeek
    # lastSaleWeek
    # lastSaleYear
    #priorSaleAmount: Replace zero with Nan
    #priorSaleDate: Split
    # priorSaleDayOfWeek
    # priorSaleWeek
    # priorSaleYear
    
    day_range = [0,6]
    week_range = [0,52]
    month_range = [0,11]
    year_range = [1900, 2020]
    transformations = [
        ('zipcode',OneHotEncoder()),
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
    lll=filter(lambda x: x[0] in features, transformations)
    return DataFrameMapper(lll)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('single_family_home_values.csv')

features = ('latitude')
df = pd.read_csv('single_family_home_values.csv')
df2=convert_dates(df)

df3=df2[~np.isnan(df2.latitude)]
# print(df3)
featurize(features).fit_transform(df3)
