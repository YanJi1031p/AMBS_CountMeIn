# required packages
import os
import xarray as xr
import numpy as np
import argparse

try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr

from utils import feature_engineering,get_fnames_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_So2Sat_pop_part1",
        type=str, help="Enter the path to So2Sat POP Part1 folder", required=True, 
        default='/p/project/hai_countmein/data/So2Sat_POP_Part1')
    
    parser.add_argument(
        "--data_path_So2Sat_pop_part2", required=True,
        type=str, help="Enter the path to So2Sat POP Part2 folder", 
        default='/p/project/hai_countmein/data/So2Sat_POP_Part2')
    
    parser.add_argument(
        "--mode", required=True,type=str,help="Enter the train or test process",default="test")

    args = parser.parse_args()
    
    all_patches_mixed_part1 = args.data_path_So2Sat_pop_part1
    all_patches_mixed_part2 = args.data_path_So2Sat_pop_part2
    mode = args.mode

    all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, mode)   # path to train folder
    all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, mode)   # path to train folder
    
    print('>>>>>>>>>> fatching samples')
    # load all the files and their corresponding population count and class for "sen2_rgb_autumn" data in "train" folder
    X_train_sen2_rgb_autumn,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_autumn')


    # load all the files and their corresponding population count and class for "sen2_rgb_summer" data in "train" folder
    X_train_sen2_rgb_summer,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_summer')


    # load all the files and their corresponding population count and class for "sen2_rgb_spring" data in "train" folder
    X_train_sen2_rgb_spring,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_spring')


    # load all the files and their corresponding population count and class for "sen2_rgb_winter" data in "train" folder
    X_train_sen2_rgb_winter,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_winter')


    # load all the files and their corresponding population count and class for "viirs" data in "train" folder
    X_train_viirs,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='viirs')


    # load all the files and their corresponding population count and class for "lcz" data in "train" folder
    X_train_lcz,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='lcz')


    # load all the files and their corresponding population count and class for "lu" data in "train" folder
    X_train_lu,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='lu')


    # load all the files and their corresponding population count and class for "dem" data in "train" folder
    X_train_dem, y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part2, data='dem')


    # load all the files and their corresponding population count and class for "osm" data in "train" folder
    X_train_osm, y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='osm_features')

    print('All training instances are loaded')


    var_list = ['X_train_sen2_rgb_spring', 'X_train_sen2_rgb_summer',
                'X_train_sen2_rgb_autumn', 'X_train_sen2_rgb_winter',
                'X_train_viirs', 'X_train_lcz',
                'X_train_lu', 'X_train_dem', 'X_train_osm']

    for ivar in range(len(var_list)):
        save_var = var_list[ivar]
        save_path = save_var+'.nc'
        if  os.path.exists(save_path) is True:
            os.remove(save_path)

        if save_var == 'X_train_osm':
            exec('n_sample,window_size,channel = '+save_var+'.shape')
            Xtrain_ds = xr.Dataset(
                eval("{'"+var_list[ivar]+"':(['n_sample','window_size','channel'],"+save_var+"),}") ,
                coords={
                         'n_sample': np.arange(n_sample),
                         'window_size': np.arange(window_size),
                         'channel': np.arange(channel),},)
            Xtrain_ds.to_netcdf(save_path)        
        else:
            exec('n_sample,height,width,channel = '+save_var+'.shape')
            Xtrain_ds = xr.Dataset(
                eval("{'"+var_list[ivar]+"':(['n_sample','height','width','channel'],"+save_var+"),}") , 
                coords={
                         'n_sample': np.arange(n_sample),
                         'height': np.arange(height),
                         'width': np.arange(width),
                         'channel': np.arange(channel),},)
            Xtrain_ds.to_netcdf(save_path)


    var_list = ['y_train_count', 'y_train_class']

    for ivar in range(len(var_list)):
        save_var = var_list[ivar]
        save_path = save_var+'.nc'
        if  os.path.exists(save_path) is True:
            os.remove(save_path)

        exec('n_sample = '+save_var+'.shape[0]')
        ytrain_ds = xr.Dataset(
            eval("{'"+save_var+"':(['n_sample'],"+save_var+"),}") ,  
            coords={'n_sample': np.arange(n_sample),},)
        ytrain_ds.to_netcdf(save_path)        

    print('All training instances are saved')



    '''
    # create features for training and testing data from So2Sat POP Part1 and So2Sat POP Part2
    feature_folder = feature_engineering(all_patches_mixed_part1)
    feature_folder = feature_engineering(all_patches_mixed_part2)

    # Perform regression, ground truth is population count (POP)
    prediction_csv = rf_regressor(feature_folder)
    '''
