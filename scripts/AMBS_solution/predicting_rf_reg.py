import glob
import os
import time
import _pickle
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

from constants import covariate_list, current_dir_path, file_name_reg, ground_truth_col_reg, ground_truth_label_col_reg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_folder", type=str, help=" the path to So2Sat POP features",
        default='/p/scratch/deepacf/deeprain/ji4/starter-pack/scripts/So2Sat_POP_features')
    parser.add_argument(
        "--model_folder", type=str, help="the path to save the trained models",
        default='/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_model')
    parser.add_argument(
        "--category_list", type=list, help="the category list for different levels of pop",
        default=[0,4,8,12,20])    
    parser.add_argument(
        "--min_fimportance_list", type=list, help="the minimum feature importance for models of different levels of pop",
        default=[0.012,0.01,0.005,0.005])    
    parser.add_argument(
        "--best_estimator_list", help="the estimator hparams for models of different levels of pop",
        default={'max_features': ['sqrt', 0.3, 0.4, 0.3],
                 'n_estimators': [1000, 1000, 1000, 500]})    

    args = parser.parse_args()
    feature_folder = args.feature_folder
    model_folder = args.model_folder
    category_list = args.category_list
    min_fimportance_list = args.min_fimportance_list
    best_estimator_list = args.best_estimator_list

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    
    # get all testing dataframe
    feature_folder_test = os.path.join(feature_folder, 'test')
    all_test_cities = glob.glob(os.path.join(feature_folder_test, '*'))
    test_df = pd.DataFrame()
    for each_city in all_test_cities:
        city_csv = glob.glob(os.path.join(each_city, '*_features.csv'))[0]  # get the feature csv
        city_df = pd.read_csv(city_csv)
        test_df = test_df.append(city_df, ignore_index=True)  # append data from all the training cities
    test_df.fillna(0, inplace=True)    
    y_test_label = test_df[ground_truth_label_col_reg]
    
    # prepare the prediction dataframe
    df_pred_complete = pd.DataFrame()

    for icate in range(len(category_list)-1):
        icate0 = category_list[icate]
        icate1 = category_list[icate+1]

        rf_features_path = os.path.join(model_folder,'rf_features_cate_%s' % str(icate))
        with open(rf_features_path, 'rb') as f:
            list_covar = _pickle.load(f)

        rf_model_path = os.path.join(model_folder,'rf_reg_cate_%s' % str(icate))
        with open(rf_model_path, 'rb') as f:
            best_estimator = _pickle.load(f)

        print("Starting training...\n")
        # Get the independent variables
        x_test = test_df[(y_test_label>=icate0)&(y_test_label<icate1)][list_covar]              
        
        # Predict on test data set
        prediction = best_estimator.predict(x_test)

        # Save the prediction
        df_pred = pd.DataFrame()
        df_pred["CITY"] = test_df[(y_test_label>=icate0)&(y_test_label<icate1)]['CITY']
        df_pred["GRD_ID"] = test_df[(y_test_label>=icate0)&(y_test_label<icate1)]['GRD_ID']
        df_pred['Predictions'] = prediction
        
        df_pred_complete = pd.concat([df_pred_complete, df_pred])

    pred_csv_path = os.path.join(model_folder, 'rf_reg_repredictions.csv')
    df_pred_complete.to_csv(pred_csv_path, index=False)
     
