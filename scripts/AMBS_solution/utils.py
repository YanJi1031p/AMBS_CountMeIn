# contains reusable helper functions
import glob
import os

import cv2
import numpy as np
import pandas as pd
import math as ma
import rasterio

import matplotlib.pyplot as plt

from rasterio.enums import Resampling

try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr

from constants import img_rows, img_cols, osm_features, current_dir_path


def raster2array(file_path, band):
    """
    :param file_path: path to the patch (raster)
    :param band: band number to read
    :return: array of raster values
    """
    raster = gdal.Open(file_path)
    band = raster.GetRasterBand(band)
    array = band.ReadAsArray()
    return array


def load_data(f_names, channels):
    """
    :param f_names: path to all the files of a data folder
    :param channels: number of channels corresponding to the data
    :return: all the instances of a data with its attributes
    """
    X = np.empty((len(f_names), img_rows, img_cols, channels))
    for i, ID in enumerate(f_names):

        # load tif file
        with rasterio.open(ID, 'r') as ds:
            image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)

        new_arr = np.empty([channels, img_rows, img_cols])

        # looping over all the channels
        for k, layer in enumerate(image):
            arr = layer
            new_arr[k] = arr
            X[i,] = np.transpose(new_arr, (1, 2, 0))
    return X


def load_osm_data(f_names, channels):
    """
    :param f_names: path to all the files of osm_features data folder
    :param channels: number of channels corresponding to the osm_features data
    :return: all the instances of osm_features data with its attributes
    """
    X = np.empty((len(f_names), osm_features, channels))
    for i, ID in enumerate(f_names):
        # load csv
        df = pd.read_csv(ID, header=None)[1]
        # remove inf and Nan values
        df = df[df.notna()]
        df_array = np.array(df)
        df_array[df_array == np.inf] = 0

        new_arr = np.empty([channels, osm_features])

        new_arr[0] = df_array

        X[i,] = np.transpose(new_arr, (1, 0))
    return X


def get_fnames_labels(folder_path, data):
    """
    :param folder_path: path to so2sat sub folder test/train
    :param data: name of the data folder, ex: 'lcz', 'lu', ...
    :return: all the instances of a data with its attributes and labels (population count & class) of each instance
    """
    city_folders = glob.glob(os.path.join(folder_path, "*"))  # list all the cities in folder_path
    f_names_all = np.array([])  # file names
    c_labels_all = np.array([])  # class labels
    p_count_all = np.array([])  # population counts
    for each_city in city_folders:
        data_path = os.path.join(each_city, data)  # path to the specifies data folder
        if data == 'dem':  # for dem data also, load the csv from So2Sat POP Part 1
            csv_path = os.path.join(each_city.replace('Part2', 'Part1'), each_city.split(os.sep)[-1:][0] + '.csv')
        else:
            csv_path = os.path.join(each_city, each_city.split(os.sep)[-1:][0] + '.csv')  # path to the cvs file of
            # the city

        city_df = pd.read_csv(csv_path)  # read csv as dataframe
        ids = city_df['GRD_ID']  # get the id of each patch
        pop = city_df['POP']  # corresponding pop count
        classes = city_df['Class']  # corresponding Class
        classes_str = [str(x) for x in classes]
        classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
        for index in range(0, len(classes_paths)):
            if data == 'osm_features':  # osm features ends with '.csv'
                f_names = [classes_paths[index] + ids[index] + '_' + data + '.csv']  # creating full path for each id
            else:
                f_names = [classes_paths[index] + ids[index] + '_' + data + '.tif']  # creating full path for each id
            f_names_all = np.append(f_names_all, f_names, axis=0)  # append file names together
            pop_count = [pop[index]]
            p_count_all = np.append(p_count_all, pop_count, axis=0)  # append pop count together
            class_label = [classes[index]]
            c_labels_all = np.append(c_labels_all, class_label, axis=0)  # append class labels together

    if data.__contains__('sen2'):
        X = load_data(f_names_all, channels=3)  # load the data for sentinel-2 files

    if data == 'viirs' or data == 'lcz' or data == "dem":
        X = load_data(f_names_all, channels=1)  # load the data for viirs, lcz, dem files

    if data == 'lu':
        X = load_data(f_names_all, channels=4)  # load the data for lu files

    if data == 'osm_features':  # load the data for osm features files
        X = load_osm_data(f_names_all, channels=1)

    return X, p_count_all, c_labels_all

def get_test_fnames_labels(folder_path, data):
    """
    :param folder_path: path to so2sat sub folder test/train
    :param data: name of the data folder, ex: 'lcz', 'lu', ...
    :return: all the instances of a data with its attributes and labels (population count & class) of each instance
    """
    city_folders = glob.glob(os.path.join(folder_path, "*"))  # list all the cities in folder_path
    f_names_all = []  # file names
    for each_city in city_folders:
        print('>>>>>>>>>>>> each_city: {}'.format(each_city))
        data_path = os.path.join(each_city, data)  # path to the specifies data folder
        
        all_classes = glob.glob(os.path.join(data_path, '*'))  # get all class folders in data folder
        for each_class in all_classes:
            class_patches = glob.glob(os.path.join(each_class, '*'))
            for x in class_patches:
                f_names_all.append(x)  # get list of all the city patches

    ids_list, city_names_list = get_id_response_var_test(f_names_all)
        
    if data.__contains__('sen2'):
        X = load_data(f_names_all, channels=3)  # load the data for sentinel-2 files

    if data == 'viirs' or data == 'lcz' or data == "dem":
        X = load_data(f_names_all, channels=1)  # load the data for viirs, lcz, dem files

    if data == 'lu':
        X = load_data(f_names_all, channels=4)  # load the data for lu files

    if data == 'osm_features':  # load the data for osm features files
        X = load_osm_data(f_names_all, channels=1)

    return X, city_names_list, ids_list

def get_id_response_var_test(all_patches):
    """
    :param all_patches: list of all patches
    :return: list of grid ids and their corresponding city list
    """
    id_list = []
    city_name_list = []
    for each_patch in all_patches:
        id = each_patch.split(os.sep)[-1].rsplit('_')[0]  # ID of the grid cell
        city_name = each_patch.split(os.sep)[-4]
        id_list.append(id)
        city_name_list.append(city_name)
    return id_list, city_name_list


def get_id_response_var_train(all_patches, city_df):
    """
    :param all_patches: list of all patches
    :param city_df: data frame for the city
    :return: grid id, population class, population count, population density, log (population density)
    """
    id_list = []
    city_name_list = []
    class_list = []
    pop_count = []
    pop_dens = []
    log_pop_dens = []
    for each_patch in all_patches:
        id = os.path.split(each_patch)[1].rsplit('_')[0]  # ID of the grid cell
        id_list.append(id)
        city_name = each_patch.split(os.sep)[-4]
        city_name_list.append(city_name)
        id_index = city_df.index[city_df['GRD_ID'] == id].tolist()[0]
        pop = city_df['POP'][id_index]  # get the absolute population count
        pop_den = pop / 1000000  # population density per 1000,000 m-sq (1km-sq)
        if pop_den == 0:
            log_pop_den = 0
        else:
            log_pop_den = ma.log(pop_den)  # calculating log of population density
        pop_count.append(pop)
        pop_dens.append(pop_den)
        log_pop_dens.append(log_pop_den)
        class_patch = each_patch.split(os.sep)[-2].rsplit('_')[1]  # corresponding class for the patch
        class_list.append(class_patch)
    return id_list, city_name_list, class_list, pop_count, pop_dens, log_pop_dens


def mean_med_std_max_min(band):
    """
    :param band: r, g, b band od sen2 patch
    :return: mean, median, std, max, min of sen2 patch
    """
    sen2_mean_band = np.mean(band)
    sen2_med_band = np.median(band)
    sen2_std_band = np.std(band)
    sen2_max_band = np.max(band)
    sen2_min_band = np.min(band)
    return sen2_mean_band, sen2_med_band, sen2_std_band, sen2_max_band, sen2_min_band


def sen2_features(all_patches):
    """
    :param all_patches: list of all patches
    :return: mean, median, std, max, min features for each r, g, b bands (5 X 3 = 15 features)
    """
    # sen2 feature lists
    sen2_mean_r_feat = []
    sen2_mean_g_feat = []
    sen2_mean_b_feat = []
    sen2_med_r_feat = []
    sen2_med_g_feat = []
    sen2_med_b_feat = []
    sen2_std_r_feat = []
    sen2_std_g_feat = []
    sen2_std_b_feat = []
    sen2_max_r_feat = []
    sen2_max_g_feat = []
    sen2_max_b_feat = []
    sen2_min_r_feat = []
    sen2_min_g_feat = []
    sen2_min_b_feat = []

    # iterate over each sen2 patch
    for each_patch in all_patches:
        sen2_array = cv2.imread(each_patch)  # read the patch
        r, g, b = sen2_array[:, :, 0], sen2_array[:, :, 1], sen2_array[:, :, 2]  # get the r, g, b bands
        # get features for r band
        sen2_mean_r, sen2_med_r, sen2_std_r, sen2_max_r, sen2_min_r = mean_med_std_max_min(band=r)
        # get features for g band
        sen2_mean_g, sen2_med_g, sen2_std_g, sen2_max_g, sen2_min_g = mean_med_std_max_min(band=g)
        # get features for b band
        sen2_mean_b, sen2_med_b, sen2_std_b, sen2_max_b, sen2_min_b = mean_med_std_max_min(band=b)
        # list of sen2 mean feature for r,g,b bands
        sen2_mean_r_feat.append(sen2_mean_r)
        sen2_mean_g_feat.append(sen2_mean_g)
        sen2_mean_b_feat.append(sen2_mean_b)
        # list of sen2 median feature for r,g,b bands
        sen2_med_r_feat.append(sen2_med_r)
        sen2_med_g_feat.append(sen2_med_g)
        sen2_med_b_feat.append(sen2_med_b)
        # list of sen2 std feature for r,g,b bands
        sen2_std_r_feat.append(sen2_std_r)
        sen2_std_g_feat.append(sen2_std_g)
        sen2_std_b_feat.append(sen2_std_b)
        # list of sen2 max feature for r,g,b bands
        sen2_max_r_feat.append(sen2_max_r)
        sen2_max_g_feat.append(sen2_max_g)
        sen2_max_b_feat.append(sen2_max_b)
        # list of sen2 min feature for r,g,b bands
        sen2_min_r_feat.append(sen2_min_r)
        sen2_min_g_feat.append(sen2_min_g)
        sen2_min_b_feat.append(sen2_min_b)

    return sen2_mean_r_feat, sen2_mean_g_feat, sen2_mean_b_feat, sen2_med_r_feat, sen2_med_g_feat, sen2_med_b_feat, \
           sen2_std_r_feat, sen2_std_g_feat, sen2_std_b_feat, sen2_max_r_feat, sen2_max_g_feat, sen2_max_b_feat, \
           sen2_min_r_feat, sen2_min_g_feat, sen2_min_b_feat


def average_mean_features(file_path, band):
    """
    :param file_path: path to patch file
    :param band: band number to read, required for multi band data
    :return: mean and max of the patch
    """
    raster_array = raster2array(file_path, band)
    raster_mean = np.mean(raster_array)
    raster_max = np.max(raster_array)
    return raster_mean, raster_max


def lu_features(file_path, band):
    """
    :param file_path: path to patch file
    :param band: band number to read, required for multi band data
    :return: total area of a patch that belongs to a particular band
    """
    lu_array = raster2array(file_path, band)
    lu_total_area = np.sum(lu_array)
    return lu_total_area


def feature_engineering(all_patches_mixed_path):
    """
    Creates features csv for each city, named city_name_features.csv
    :param all_patches_mixed_path: path to the folder that contains the cities to process
    :return: None
    """
    # preparing features for part 1 of dataset
    if all_patches_mixed_path.__contains__("Part1"):
        print('\nPreparing features for So2sat Part1')
        feature_folder = os.path.join(current_dir_path, 'So2Sat_POP_features')
        if not os.path.exists(feature_folder):
            os.mkdir(feature_folder)
        feature_folder_train = os.path.join(feature_folder, 'train')
        if not os.path.exists(feature_folder_train):
            os.mkdir(feature_folder_train)
        feature_folder_test = os.path.join(feature_folder, 'test')
        if not os.path.exists(feature_folder_test):
            os.mkdir(feature_folder_test)

        all_folders = glob.glob(os.path.join(all_patches_mixed_path, '*'))
        for each_folder in all_folders:
            all_cities = glob.glob(os.path.join(each_folder, '*'))
            for each_city in all_cities:
                # declare lists for input data source
                lu_1_feat = []
                lu_2_feat = []
                lu_3_feat = []
                lu_4_feat = []
                lcz_feat = []
                viirs_mean_feat = []
                viirs_max_feat = []
                osm_feat = []
                HINT_CLASS = []

                city_name = os.path.split(each_city)[1]  # get the name of the city from the city path

                feature_folder_city = os.path.join(feature_folder, each_city.split(os.sep)[-2], city_name)
                if not os.path.exists(feature_folder_city):
                    os.mkdir(feature_folder_city)

                feature_csv_file = os.path.join(feature_folder_city, city_name + '_features.csv')  # create feature csv for city
                all_data = glob.glob(os.path.join(each_city, '*'))  # get all the data folders

                lu_path = glob.glob(os.path.join(each_city, 'lu'))
                all_classes = glob.glob(os.path.join(lu_path[0], '*'))  # get all class folders in data folder
                for each_class in all_classes:
                    hint_c = int(os.path.split(each_class)[1].split('_')[-1])
                    class_patches = glob.glob(os.path.join(each_class, '*'))
                    for x in class_patches:
                        HINT_CLASS.append(hint_c)

                for each_data in all_data:  # for each data folder in a city
                    all_patches = []  # list to all patches
                    if each_data.endswith('.csv'):  # skip the csv file, get only data folders
                        # skip the file
                        continue
                    all_classes = glob.glob(os.path.join(each_data, '*'))  # get all class folders in data folder
                    for each_class in all_classes:
                        class_patches = glob.glob(os.path.join(each_class, '*'))
                        for x in class_patches:
                            all_patches.append(x)  # get list of all the city patches

                    if each_data.endswith('lu'):  # process lu data
                        for each_patch in all_patches:
                            area_lu_1 = lu_features(each_patch,
                                                    band=1)  # area that belongs to band 1 (commercial) of lu patch
                            area_lu_2 = lu_features(each_patch,
                                                    band=2)  # area that belongs to band 2 (industrial) of lu patch
                            area_lu_3 = lu_features(each_patch,
                                                    band=3)  # area that belongs to band 3 (residential) of lu patch
                            area_lu_4 = lu_features(each_patch,
                                                    band=4)  # area that belongs to band 4 (other) of lu patch
                            lu_1_feat.append(area_lu_1)  # lu band 1 area feature list
                            lu_2_feat.append(area_lu_2)  # lu band 2 area feature list
                            lu_3_feat.append(area_lu_3)  # lu band 3 area feature list
                            lu_4_feat.append(area_lu_4)  # lu band 4 area feature list

                    if each_data.endswith('lcz'):  # process lcz data
                        for each_patch in all_patches:
                            lcz_array = raster2array(each_patch, 1)
                            lcz_class = np.argmax(
                                np.bincount(lcz_array.flatten()))  # get the majority lcz class of the patch
                            lcz_feat.append(lcz_class)  # majority lcz class feature list

                    if each_data.endswith('viirs'):  # process nightlights data
                        for each_patch in all_patches:
                            viirs_mean, viirs_max = average_mean_features(each_patch,
                                                                          band=1)  # get mean and max of viirs patch
                            viirs_mean_feat.append(viirs_mean)  # viirs mean feature list
                            viirs_max_feat.append(viirs_max)  # viirs max feature list

                    if each_data.endswith('autumn'):  # process sen2 autumn data
                        sen2_aut_mean_r_feat, sen2_aut_mean_g_feat, sen2_aut_mean_b_feat, sen2_aut_med_r_feat, \
                        sen2_aut_med_g_feat, sen2_aut_med_b_feat, sen2_aut_std_r_feat, sen2_aut_std_g_feat, \
                        sen2_aut_std_b_feat, sen2_aut_max_r_feat, sen2_aut_max_g_feat, sen2_aut_max_b_feat, \
                        sen2_aut_min_r_feat, sen2_aut_min_g_feat, sen2_aut_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('spring'):  # process sen2 spring data
                        sen2_spr_mean_r_feat, sen2_spr_mean_g_feat, sen2_spr_mean_b_feat, sen2_spr_med_r_feat, \
                        sen2_spr_med_g_feat, sen2_spr_med_b_feat, sen2_spr_std_r_feat, sen2_spr_std_g_feat, \
                        sen2_spr_std_b_feat, sen2_spr_max_r_feat, sen2_spr_max_g_feat, sen2_spr_max_b_feat, \
                        sen2_spr_min_r_feat, sen2_spr_min_g_feat, sen2_spr_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('summer'):  # process sen2 summer data
                        sen2_sum_mean_r_feat, sen2_sum_mean_g_feat, sen2_sum_mean_b_feat, sen2_sum_med_r_feat, \
                        sen2_sum_med_g_feat, sen2_sum_med_b_feat, sen2_sum_std_r_feat, sen2_sum_std_g_feat, \
                        sen2_sum_std_b_feat, sen2_sum_max_r_feat, sen2_sum_max_g_feat, sen2_sum_max_b_feat, \
                        sen2_sum_min_r_feat, sen2_sum_min_g_feat, sen2_sum_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('winter'):  # process sen2 winter data
                        sen2_win_mean_r_feat, sen2_win_mean_g_feat, sen2_win_mean_b_feat, sen2_win_med_r_feat, \
                        sen2_win_med_g_feat, sen2_win_med_b_feat, sen2_win_std_r_feat, sen2_win_std_g_feat, \
                        sen2_win_std_b_feat, sen2_win_max_r_feat, sen2_win_max_g_feat, sen2_win_max_b_feat, \
                        sen2_win_min_r_feat, sen2_win_min_g_feat, sen2_win_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('osm_features'):  # process the osm data
                        for each_patch in all_patches:
                            osm_features = pd.read_csv(each_patch, header=None)  # read the osm feature csv file
                            osm_features = osm_features.dropna()  # drop the NA fields
                            osm_features = osm_features.T  # take the transpose to (2,56)
                            all_keys = osm_features.iloc[0].tolist()  # get all the keys
                            values = osm_features.iloc[1].tolist()  # get all the corresponding values for the keys
                            values = [0 if x == np.inf else x for x in values]  # remove inf values
                            values = [0 if x == np.nan else x for x in values]  # remove nan values
                            osm_feat.append(values)  # append osm features for each patch

                        df_osm = pd.DataFrame(osm_feat, columns=all_keys)  # data frame for osm features
                        df_rest = pd.DataFrame()  # initialize data frame for a city
                        # add all the features to data frame
                        city_csv_file = os.path.join(each_city, city_name + '.csv')  # get the city's csv
                        id_list, city_list = get_id_response_var_test(all_patches)
                        df_rest['CITY'] = city_list
                        df_rest['GRD_ID'] = id_list
                        df_rest['HINT_CLASS'] = HINT_CLASS 

                        if each_folder.__contains__('train'):
                            city_df = pd.read_csv(city_csv_file)  # data frame for the city
                            print('city_csv_file', city_csv_file)
                            id_list, city_list, class_list, pop_count, pop_dens, log_pop_dens = get_id_response_var_train(
                                all_patches, city_df)
                            df_rest['POP'] = pop_count
                            df_rest['POP_DENS'] = pop_dens
                            df_rest['LOG_POP_DENS'] = log_pop_dens

                # Features
                df_rest['LCZ_CL'] = lcz_feat
                df_rest['LU_1_A'] = lu_1_feat
                df_rest['LU_2_A'] = lu_2_feat
                df_rest['LU_3_A'] = lu_3_feat
                df_rest['LU_4_A'] = lu_4_feat
                df_rest['VIIRS_MEAN'] = viirs_mean_feat
                df_rest['VIIRS_MAX'] = viirs_max_feat

                df_rest['SEN2_AUT_MEAN_R'] = sen2_aut_mean_r_feat
                df_rest['SEN2_AUT_MEAN_G'] = sen2_aut_mean_g_feat
                df_rest['SEN2_AUT_MEAN_B'] = sen2_aut_mean_b_feat
                df_rest['SEN2_AUT_MED_R'] = sen2_aut_med_r_feat
                df_rest['SEN2_AUT_MED_G'] = sen2_aut_med_g_feat
                df_rest['SEN2_AUT_MED_B'] = sen2_aut_med_b_feat
                df_rest['SEN2_AUT_STD_R'] = sen2_aut_std_r_feat
                df_rest['SEN2_AUT_STD_G'] = sen2_aut_std_g_feat
                df_rest['SEN2_AUT_STD_B'] = sen2_aut_std_b_feat
                df_rest['SEN2_AUT_MAX_R'] = sen2_aut_max_r_feat
                df_rest['SEN2_AUT_MAX_G'] = sen2_aut_max_g_feat
                df_rest['SEN2_AUT_MAX_B'] = sen2_aut_max_b_feat
                df_rest['SEN2_AUT_MIN_R'] = sen2_aut_min_r_feat
                df_rest['SEN2_AUT_MIN_G'] = sen2_aut_min_g_feat
                df_rest['SEN2_AUT_MIN_B'] = sen2_aut_min_b_feat

                df_rest['SEN2_SPR_MEAN_R'] = sen2_spr_mean_r_feat
                df_rest['SEN2_SPR_MEAN_G'] = sen2_spr_mean_g_feat
                df_rest['SEN2_SPR_MEAN_B'] = sen2_spr_mean_b_feat
                df_rest['SEN2_SPR_MED_R'] = sen2_spr_med_r_feat
                df_rest['SEN2_SPR_MED_G'] = sen2_spr_med_g_feat
                df_rest['SEN2_SPR_MED_B'] = sen2_spr_med_b_feat
                df_rest['SEN2_SPR_STD_R'] = sen2_spr_std_r_feat
                df_rest['SEN2_SPR_STD_G'] = sen2_spr_std_g_feat
                df_rest['SEN2_SPR_STD_B'] = sen2_spr_std_b_feat
                df_rest['SEN2_SPR_MAX_R'] = sen2_spr_max_r_feat
                df_rest['SEN2_SPR_MAX_G'] = sen2_spr_max_g_feat
                df_rest['SEN2_SPR_MAX_B'] = sen2_spr_max_b_feat
                df_rest['SEN2_SPR_MIN_R'] = sen2_spr_min_r_feat
                df_rest['SEN2_SPR_MIN_G'] = sen2_spr_min_g_feat
                df_rest['SEN2_SPR_MIN_B'] = sen2_spr_min_b_feat

                df_rest['SEN2_SUM_MEAN_R'] = sen2_sum_mean_r_feat
                df_rest['SEN2_SUM_MEAN_G'] = sen2_sum_mean_g_feat
                df_rest['SEN2_SUM_MEAN_B'] = sen2_sum_mean_b_feat
                df_rest['SEN2_SUM_MED_R'] = sen2_sum_med_r_feat
                df_rest['SEN2_SUM_MED_G'] = sen2_sum_med_g_feat
                df_rest['SEN2_SUM_MED_B'] = sen2_sum_med_b_feat
                df_rest['SEN2_SUM_STD_R'] = sen2_sum_std_r_feat
                df_rest['SEN2_SUM_STD_G'] = sen2_sum_std_g_feat
                df_rest['SEN2_SUM_STD_B'] = sen2_sum_std_b_feat
                df_rest['SEN2_SUM_MAX_R'] = sen2_sum_max_r_feat
                df_rest['SEN2_SUM_MAX_G'] = sen2_sum_max_g_feat
                df_rest['SEN2_SUM_MAX_B'] = sen2_sum_max_b_feat
                df_rest['SEN2_SUM_MIN_R'] = sen2_sum_min_r_feat
                df_rest['SEN2_SUM_MIN_G'] = sen2_sum_min_g_feat
                df_rest['SEN2_SUM_MIN_B'] = sen2_sum_min_b_feat

                df_rest['SEN2_WIN_MEAN_R'] = sen2_win_mean_r_feat
                df_rest['SEN2_WIN_MEAN_G'] = sen2_win_mean_g_feat
                df_rest['SEN2_WIN_MEAN_B'] = sen2_win_mean_b_feat
                df_rest['SEN2_WIN_MED_R'] = sen2_win_med_r_feat
                df_rest['SEN2_WIN_MED_G'] = sen2_win_med_g_feat
                df_rest['SEN2_WIN_MED_B'] = sen2_win_med_b_feat
                df_rest['SEN2_WIN_STD_R'] = sen2_win_std_r_feat
                df_rest['SEN2_WIN_STD_G'] = sen2_win_std_g_feat
                df_rest['SEN2_WIN_STD_B'] = sen2_win_std_b_feat
                df_rest['SEN2_WIN_MAX_R'] = sen2_win_max_r_feat
                df_rest['SEN2_WIN_MAX_G'] = sen2_win_max_g_feat
                df_rest['SEN2_WIN_MAX_B'] = sen2_win_max_b_feat
                df_rest['SEN2_WIN_MIN_R'] = sen2_win_min_r_feat
                df_rest['SEN2_WIN_MIN_G'] = sen2_win_min_g_feat
                df_rest['SEN2_WIN_MIN_B'] = sen2_win_min_b_feat

                df = pd.concat([df_rest, df_osm], axis=1)  # appending the rest of features and osm features
                df.to_csv(feature_csv_file, index=False)  # save the features to csv files
                print("City {} finished".format(city_name))
        print('All cities processed for So2Sat POP Part 1 \n')

    else:
        print('Preparing features for So2sat Part2')
        all_folders = glob.glob(os.path.join(all_patches_mixed_path, '*'))
        for each_folder in all_folders:
            all_patches_mixed_path_part1 = each_folder.replace('Part2', 'Part1')
            all_cities = glob.glob(os.path.join(each_folder, '*'))
            for each_city in all_cities:
                # declare lists for input data source
                dem_mean_feat = []
                dem_max_feat = []

                city_name = os.path.split(each_city)[1]  # get the name of the city from the city path
                feature_folder = os.path.join(current_dir_path, 'So2Sat_POP_features')

                feature_folder_city = os.path.join(feature_folder, each_city.split(os.sep)[-2], city_name)
                feature_csv_file = glob.glob(os.path.join(feature_folder_city, '*.csv'))[0]
                all_data = glob.glob(os.path.join(each_city, '*'))  # get all the data folders

                for each_data in all_data:  # for each data folder in a city
                    all_patches = []  # list to all patches
                    if each_data.endswith('.csv'):  # skip the csv file, get only data folders
                        # skip the file
                        continue
                    all_classes = glob.glob(os.path.join(each_data, '*'))  # get all class folders in data folder
                    for each_class in all_classes:
                        class_patches = glob.glob(os.path.join(each_class, '*'))
                        for x in class_patches:
                            all_patches.append(x)  # get list of all the city patches

                    if each_data.endswith('dem'):  # process dem data
                        for each_patch in all_patches:
                            dem_mean, dem_max = average_mean_features(each_patch,
                                                                      band=1)  # get mean and max of dem patch
                            dem_mean_feat.append(dem_mean)  # dem mean feature list
                            dem_max_feat.append(dem_max)  # dem max feature list

                df_rest = pd.DataFrame()  # initialize data frame for a city

                if os.path.isfile(feature_csv_file):
                    df_rest['DEM_MEAN'] = dem_mean_feat
                    df_rest['DEM_MAX'] = dem_max_feat
                    df_part1 = pd.read_csv(feature_csv_file)
                    df = pd.concat([df_part1, df_rest], axis=1)  # appending the rest of features and osm features
                    df.to_csv(feature_csv_file, index=False)  # save the features to csv files
                    print("City {} finished".format(city_name))
                else:
                    print('No feature file found in part1. Please check')
                    
        print('All cities processed for So2Sat POP Part 2 \n')
        return feature_folder

def validation_reg(pred_csv_path, validation_csv_path, all_patches_mixed_test_part1):
    """
    :param pred_csv_path: Path to csv file, has saved predictions and expected values for test data
    :param oob_score: Final OOB score of the model
    :param validation_csv_path: path to new csv which will contain the metric for further evaluations
    """
    all_cities = glob.glob(os.path.join(all_patches_mixed_test_part1, '*'))
    for each_city in all_cities:
        ref_csv_path = glob.glob(os.path.join(each_city, '*.csv'))[0]

    log_list = []
    df_pred = pd.read_csv(pred_csv_path)
    df_ref = pd.read_csv(ref_csv_path)
    df = df_pred.merge(df_ref, on=['CITY', 'GRD_ID'], how='outer', suffixes=['_pred', '_ref'])
    output_log_validation = validation_csv_path.replace('.csv', '_log.txt')
    # Compute error
    df['error'] = df['Predictions'] - df['POP']
    # Compute squared error raster
    df['sqerror'] = df['error'] ** 2
    # Compute absolute error raster
    df['abserror'] = abs(df['error'])

    # Compute overall validation statistics #
    rmse = ma.sqrt(round(df['sqerror'].mean(), 2))  # Compute RMSE (Root mean squared error)
    mean_ref = df['POP'].mean()  # Compute mean reference population per admin unit
    prct_rmse = (rmse / mean_ref) * 100  # Compute %RMSE
    MAE = df['abserror'].mean()  # Compute MAE (Mean absolute error)
    TAE = df['abserror'].sum()  # Compute TAE (Total absolute error)
    POPTOT = df['POP'].sum()  # Compute Total reference population
    PREDTOT = df['Predictions'].sum()  # Compute Total predicted population
    corr = np.corrcoef(df['POP'], df['Predictions'])[0, 1]  # Get correlation value
    r_squared = (corr ** 2)  # Get r-squared value

    # Outputs print and log
    log = ""
    log += "Total reference population = %s \n" % round(POPTOT, 1)
    log += "Total predicted population = %s \n" % round(PREDTOT, 2)
    log += '\n'
    log += "Mean absolute error of prediction (MAE) = %s \n" % round(MAE, 3)
    log += "Root mean squared error of prediction (RMSE) = %s \n" % round(rmse, 3)
    log += "Root mean squared error of prediction in percentage (Percent_RMSE) = %s \n" % round(prct_rmse, 3)
    log += "Total absolute error (TAE) = %s \n" % round(TAE, 3)
    log += "R squared = %s \n" % round(r_squared, 3)
    log_list.extend([[round(MAE, 3), round(prct_rmse, 3), round(r_squared, 3)]])
    print("Mean absolute error of prediction (MAE): ", round(MAE, 3))
    print("Root mean squared error of prediction (RMSE): ", round(rmse, 3))
    print("R squared : ", round(r_squared, 3))
    print("\nPlease check detailed log at {} \n ".format(output_log_validation))
    fout = open(output_log_validation, 'w')
    fout.write(log)
    fout.close()

    predicted_value = df['Predictions']  # get the predictions
    true_value = df['POP']  # get the actual values
    # plot predicted vs actual values
    plt.figure(figsize=(10, 10))
    plt.scatter(true_value, predicted_value, c='crimson')
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title('Predictions vs Actual Values')
    plt.axis('equal')
    path_plot = validation_csv_path.replace('.csv', '_predvsref.png')
    plt.savefig(path_plot, bbox_inches='tight', dpi=400)
    print('Finished regression')


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
