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
from xgboost import XGBRegressor
#from sklearn.ensemble import LGBMRegressor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
def prepare_test_train_validation(df, split_by_zipcode=False, seed=1234, 
                                  train_percent=0.6, validate_percent=0.2):
    print("Splitting test train and validatiaon")
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
                zero_to_nan_fields=None):
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
    
    # Remove outlier values
    if remove_outliers:
        None
    df = convert_dates(df, date_fields)

    return df
    

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
    for field in date_fields:
        d = pd.Series(pd.to_datetime(df[field],format='%Y-%m-%d'))
        
        df[field+'DayOfWeek'] = d.dt.dayofweek
        df[field+'WeekOfYear'] = d.dt.weekofyear
        df[field+'Month'] = d.dt.month
        df[field+'Year'] = d.dt.year

    return remove_not_useful_fields(df, date_fields)
        
        
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


def abs_mean_relative_error(y,y_pred):
        return np.mean(np.abs(y-y_pred)/(y))


df = pd.read_csv('single_family_home_values.csv')


print('Preparing Data: cleaning data')
not_useful_fields = ['id','city','state', 'address']
date_fields = ['lastSaleDate', 'priorSaleDate']
required_fields = ['latitude','longitude','zipcode','bedrooms',
                   'bathrooms','rooms','squareFootage',
                   'lotSize','yearBuilt',
                   'lastSaleDate','estimated_value']
zero_to_nan_fields = None
remove_outliers=False
df2 = prepare_data(df, not_useful_fields=not_useful_fields, 
        date_fields=date_fields,
        remove_outliers=remove_outliers,
        required_fields=required_fields,
        zero_to_nan_fields=zero_to_nan_fields,
        )

print('Preparing Data: splitting test/train/validation')
df_train, df_test, df_validation = train_validate_test_split(df2)

X_test = df_test[df_train.columns.drop('estimated_value')]
y_test = np.log(df_test['estimated_value'] + 1)


'''
#features = ('latitude', 'longitude')
#featurize(features).fit_transform(df2)
print("Setting up pipeline")
features = ('latitude')
features = required_fields

#: Note imputer will strip away column heads, it has to be after featurize
pipeline = Pipeline([
          ('featurize', featurize(features)), 
          ('imputer', Imputer(missing_values=np.nan, strategy="mean", axis=0)),
          ('forest', RandomForestRegressor()),
            ])
print("Fitting data")
model = pipeline.fit(X = X, y = y)
print("Evaluating model")
y_pred = model.predict(X)
plt.plot(y, y_pred,'.')
y_max = np.max((y, y_pred))
plt.plot((0,y_max), (0,y_max),color='black')
plt.show()
'''
#TODO: Evaluate different feature sets. 
#TODO: error handling of input data types
#TODO: Ensure more broad testing of input data types (Nan/Zero handling)
#TODO: Improve plotting/metric evaluation
#TODO: Github repository
#TODO: Robust deployment on different machine

#from IPython import embed
#embed()
# IDEA: Cluster based on missing data!
# Ensure zipcode
