from __future__ import division, print_function

import argparse
import os
import re
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.externals import joblib
import os

#TODO: implement loggin

def main():
    args = _get_args()

    df = pd.read_csv(args.input_filename)
    df_id = df['id']
    print('Adding self scaling features')
    #: TODO: This needs to be put into the prepare_data_pipeline
    df['squareFootageOverLotSize'] = df['squareFootage']/df['lotSize']
    df['bathroomsPerRooms'] = df['bathrooms']/(df['rooms']+.01)
    df['roomsPerSquareFootage'] = df['rooms']/(df['squareFootage'])
    df['hasPriorSale'] = ~pd.isnull(df['priorSaleDate'])

    print('Loading model {} for data prep'.format(args.model_locations[0]))
    prepare_data_pipeline = joblib.load(args.model_locations[0])
    print('Loading model {} for fitting'.format(args.model_locations[1]))
    fit_data_pipeline = joblib.load(args.model_locations[1])
    x_test = prepare_data_pipeline.transform(df)
    y_test_pred = fit_data_pipeline.predict(x_test)
    
    df_pred = pd.DataFrame()
    df_pred['id'] = df_id
    df_pred['estimated_value_new'] = y_test_pred
    df_pred.set_index('id', inplace=True)
    print("Saving predictions to {}".format(args.output_filename))
    df_pred.to_csv(args.output_filename)

    return

def _get_args():
    default_model_locations = [
            'models/finalized_prepare_data_pipeline.sav',
            'models/finalized_fit_data_pipeline.sav'
            ]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = u"""
        Predict housing prices from input csv file..
        """
    parser.add_argument(
        '-i', '--input_filename',
        help='The input filename for the housing data you want to estimate prices on.',
        default='data/single_family_home_values.csv'
    )
    parser.add_argument(
        '-m', '--model_locations',
        default=default_model_locations,
        help='The different model_locations you might want to refer to'
    )
    parser.add_argument(
        '-o', '--output_filename',
        default='./predictions.csv',
        help='The ouptut filename'
    )
    
    args = parser.parse_args() 

    return args


if __name__=='__main__':
    
    main()
