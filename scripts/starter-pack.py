# required packages
import os
import argparse

try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr

from utils import feature_engineering
from rf_regression import rf_regressor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_So2Sat_pop_part1",
        type=str, help="Enter the path to So2Sat POP Part1 folder", required=True)
    
    parser.add_argument(
        "--data_path_So2Sat_pop_part2", required=True,
        type=str, help="Enter the path to So2Sat POP Part2 folder")
    
    args = parser.parse_args()
    
    all_patches_mixed_part1 = args.data_path_So2Sat_pop_part1
    all_patches_mixed_part2 = args.data_path_So2Sat_pop_part2

    all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, 'train')   # path to train folder
    all_patches_mixed_test_part1 = os.path.join(all_patches_mixed_part1, 'test')  # path to test folder
    
    all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, 'train')   # path to train folder
    all_patches_mixed_test_part2 = os.path.join(all_patches_mixed_part2, 'test')   # path to test folder

    print('\nPath to So2Sat POP Part1: ', all_patches_mixed_part1)
    print('Path to So2Sat POP Part2: ', all_patches_mixed_part2)
    
    # create features for training and testing data from So2Sat POP Part1 and So2Sat POP Part2
    feature_folder = feature_engineering(all_patches_mixed_part1)
    feature_folder = feature_engineering(all_patches_mixed_part2)

    # Perform regression, ground truth is population count (POP)
    prediction_csv = rf_regressor(feature_folder)
