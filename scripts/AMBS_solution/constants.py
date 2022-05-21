import os

# constants
img_rows = 100
img_cols = 100
osm_features = 56

# paths to the current folder
current_dir_path = os.getcwd()

# suffix for the file names
file_name_reg = 'rf_reg'
ground_truth_col_reg = 'POP'
ground_truth_label_col_reg = 'HINT_CLASS'
pred_label_col_reg = 'Predictions'

# covariates used to train the model
covariate_list = ['DEM_MEAN', 'DEM_MAX', 'LCZ_CL', 'LU_1_A', 'LU_2_A', 'LU_3_A', 'LU_4_A', 'VIIRS_MEAN', 'VIIRS_MAX',
                  'SEN2_AUT_MEAN_R', 'SEN2_AUT_MEAN_G', 'SEN2_AUT_MEAN_B', 'SEN2_AUT_MED_R', 'SEN2_AUT_MED_G',
                  'SEN2_AUT_MED_B', 'SEN2_AUT_STD_R', 'SEN2_AUT_STD_G', 'SEN2_AUT_STD_B', 'SEN2_AUT_MAX_R',
                  'SEN2_AUT_MAX_G', 'SEN2_AUT_MAX_B', 'SEN2_AUT_MIN_R', 'SEN2_AUT_MIN_G', 'SEN2_AUT_MIN_B',
                  'SEN2_SPR_MEAN_R', 'SEN2_SPR_MEAN_G', 'SEN2_SPR_MEAN_B', 'SEN2_SPR_MED_R', 'SEN2_SPR_MED_G',
                  'SEN2_SPR_MED_B', 'SEN2_SPR_STD_R', 'SEN2_SPR_STD_G', 'SEN2_SPR_STD_B', 'SEN2_SPR_MAX_R',
                  'SEN2_SPR_MAX_G', 'SEN2_SPR_MAX_B', 'SEN2_SPR_MIN_R', 'SEN2_SPR_MIN_G', 'SEN2_SPR_MIN_B',
                  'SEN2_SUM_MEAN_R', 'SEN2_SUM_MEAN_G', 'SEN2_SUM_MEAN_B', 'SEN2_SUM_MED_R', 'SEN2_SUM_MED_G',
                  'SEN2_SUM_MED_B', 'SEN2_SUM_STD_R', 'SEN2_SUM_STD_G', 'SEN2_SUM_STD_B', 'SEN2_SUM_MAX_R',
                  'SEN2_SUM_MAX_G', 'SEN2_SUM_MAX_B', 'SEN2_SUM_MIN_R', 'SEN2_SUM_MIN_G', 'SEN2_SUM_MIN_B',
                  'SEN2_WIN_MEAN_R', 'SEN2_WIN_MEAN_G', 'SEN2_WIN_MEAN_B', 'SEN2_WIN_MED_R', 'SEN2_WIN_MED_G',
                  'SEN2_WIN_MED_B', 'SEN2_WIN_STD_R', 'SEN2_WIN_STD_G', 'SEN2_WIN_STD_B', 'SEN2_WIN_MAX_R',
                  'SEN2_WIN_MAX_G', 'SEN2_WIN_MAX_B', 'SEN2_WIN_MIN_R', 'SEN2_WIN_MIN_G', 'SEN2_WIN_MIN_B', 'aerialway',
                  'aeroway', 'amenity', 'barrier', 'boundary', 'building', 'craft', 'emergency', 'geological',
                  'healthcare', 'highway', 'historic', 'landuse', 'leisure', 'man_made', 'military', 'natural',
                  'office', 'place', 'power', 'public Transport', 'railway', 'route', 'shop', 'sport', 'telecom',
                  'tourism', 'water', 'waterway', 'addr:housenumber', 'restrictions', 'other', 'n', 'm', 'k_avg',
                  'intersection_count', 'streets_per_node_avg', 'streets_per_node_counts_argmin',
                  'streets_per_node_counts_min', 'streets_per_node_counts_argmax', 'streets_per_node_counts_max',
                  'streets_per_node_proportion_argmin', 'streets_per_node_proportion_min',
                  'streets_per_node_proportion_argmax', 'streets_per_node_proportion_max', 'edge_length_total',
                  'edge_length_avg', 'street_length_total', 'street_length_avg', 'street_segments_count',
                  'node_density_km', 'intersection_density_km', 'edge_density_km', 'street_density_km', 'circuity_avg',
                  'self_loop_proportion']


# Parameters for baseline experiment
min_fimportance = 0.002

# Parameter for the Grid Search for hyperparameter optimization
param_grid = {'oob_score': [True], 'bootstrap': [True],
              'max_features': ['sqrt', 0.05, 0.1, 0.2, 0.3, 0.4],
              'n_estimators': [250, 350, 500, 750, 1000]}

# Kfold parameter
kfold = 3
n_jobs = -1
