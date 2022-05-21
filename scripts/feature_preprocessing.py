import glob
import os
import time
import _pickle

import pandas as pd

from utils import plot_feature_importance, feature_engineering
from constants import min_fimportance, kfold, n_jobs, param_grid, covariate_list, current_dir_path, file_name_reg, ground_truth_col_reg, ground_truth_label_col_reg

# required packages
import xarray as xr
import numpy as np
import argparse

try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_So2Sat_pop_part1",
        type=str, help="Enter the path to So2Sat POP Part1 folder",
        default='/p/project/hai_countmein/data/So2Sat_POP_Part1')

    parser.add_argument(
        "--data_path_So2Sat_pop_part2", 
        type=str, help="Enter the path to So2Sat POP Part2 folder",
        default='/p/project/hai_countmein/data/So2Sat_POP_Part2')

    args = parser.parse_args()

    all_patches_mixed_part1 = args.data_path_So2Sat_pop_part1
    all_patches_mixed_part2 = args.data_path_So2Sat_pop_part2

    feature_folder = feature_engineering(all_patches_mixed_part1)
    feature_folder = feature_engineering(all_patches_mixed_part2)
