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

# from utils import plot_feature_importance
from constants import covariate_list, current_dir_path, file_name_reg, ground_truth_col_reg, ground_truth_label_col_reg

def plot_feature_importance(importances, x_test, path_plot):
    """
    :param importances: array of feature importance from the model
    :param x_test: data frame for test cities
    :param path_plot: path to feature importance plot
    :return: Create and save the feature importance plot
    """
    indices = np.argsort(importances)[::-1]
    indices = indices[:12]  # get indices of only top 12 features
    x_axis = importances[indices][::-1]
    idx = indices[::-1]
    y_axis = range(len(x_axis))
    Labels = []
    for i in range(len(x_axis)):
        Labels.append(x_test.columns[idx[i]])  # get corresponding labels of the features
    y_ticks = np.arange(0, len(x_axis))
    fig, ax = plt.subplots()
    ax.barh(y_axis, x_axis)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(Labels)
    ax.set_title("Random Forest TOP 12 Important Features")
    fig.tight_layout()
    plt.savefig(path_plot, bbox_inches='tight', dpi=400)  # Export in .png file (image)

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
        default=[0,4,6,8,10,12,14,20])    
    parser.add_argument(
        "--min_fimportance_list", type=list, help="the minimum feature importance for models of different levels of pop",
        default=[0.012,0.014,0.012,0.01,0.008,0.008,0.008])    
    parser.add_argument(
        "--best_estimator_list", help="the estimator hparams for models of different levels of pop",
        default={'max_features': [0.2, 0.5, 0.6, 0.5, 0.4, 0.6, 0.6],
                 'n_estimators': [1500, 1000, 1500, 1500, 750, 1500, 1250]})    

    args = parser.parse_args()
    feature_folder = args.feature_folder
    model_folder = args.model_folder
    category_list = args.category_list
    min_fimportance_list = args.min_fimportance_list
    best_estimator_list = args.best_estimator_list

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    print("Starting regression")
    # get all training cities
    feature_folder_train = os.path.join(feature_folder, 'train')
    all_train_cities = glob.glob(os.path.join(feature_folder_train, '*'))
    
    # prepare the training dataframe
    training_df = pd.DataFrame()
    for each_city in all_train_cities:
        city_csv = glob.glob(os.path.join(each_city, '*_features.csv'))[0]  # get the feature csv
        city_df = pd.read_csv(city_csv)
        training_df = training_df.append(city_df, ignore_index=True)  # append data from all the training cities
    training_df.fillna(0, inplace=True)    
    y_label = training_df[ground_truth_label_col_reg]
    num_classes = len(np.unique(y_label))
    
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
        # Get the dependent variables
        y = training_df[(y_label>=icate0)&(y_label<icate1)][ground_truth_col_reg]
        # Get the independent variables
        x = training_df[(y_label>=icate0)&(y_label<icate1)][covariate_list]

        print("Starting training...\n")
        # Initialize the model
        rfmodel = RandomForestRegressor(n_estimators=500, oob_score=True, max_features='auto', n_jobs=-1,
                                        random_state=0)  # random_state is fixed to allow exact replication
        min_fimportance = min_fimportance_list[icate]
        sel = SelectFromModel(rfmodel, threshold=min_fimportance)
        fited = sel.fit(x, y)
        feature_idx = fited.get_support()  # Get list of T/F for covariates for which OOB score is upper the threshold
        list_covar = list(x.columns[feature_idx])  # Get list of covariates with the selected features

        x = fited.transform(x) # Update the dataframe with the selected features only
        best_rfmodel = RandomForestRegressor(n_estimators=best_estimator_list['n_estimators'][icate], 
                                               max_features=best_estimator_list['max_features'][icate],
                                                oob_score=True, n_jobs=-1, random_state=0)
        best_estimator = best_rfmodel.fit(x, y)

        rf_features_path = os.path.join(model_folder,'rf_features_cate_%s' % str(icate))
        if os.path.exists(rf_features_path):
            os.remove(rf_features_path)

        # save the selected features
        with open(rf_features_path, 'wb') as f:
            _pickle.dump(list_covar, f)
            f.close()

        rf_model_path = os.path.join(model_folder,'rf_reg_cate_%s' % str(icate))
        if os.path.exists(rf_model_path):
            os.remove(rf_model_path)

        # save the best regressor
        with open(rf_model_path, 'wb') as f:
            _pickle.dump(best_estimator, f)
            f.close()

        print("Starting predicting...\n")
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

        # Feature importances
        print("Creation of feature importance plot...\n")
        importances = best_estimator.feature_importances_  # Save feature importances from the model
        path_plot = os.path.join(model_folder, "rf_feature_importance_cate_%s" % str(icate))  # path to saved plot
        plot_feature_importance(importances, x_test, path_plot)

    pred_csv_path = os.path.join(model_folder, 'rf_reg_predictions.csv')
    df_pred_complete.to_csv(pred_csv_path, index=False)
    # return pred_csv_path
     
