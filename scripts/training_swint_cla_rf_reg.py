import glob
import os
import time
import _pickle
import numpy as np
import math as ma
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import gc
import xarray as xr
import random
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

from constants import covariate_list, current_dir_path, file_name_reg
from constants import ground_truth_col_reg, ground_truth_label_col_reg, pred_label_col_reg


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

def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
    
class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

def NormMeanStd(array,dim=0):
    array_mean = np.nanmean(array,dim)
    array_std = np.nanstd(array,dim)
    array_nor = (array-array_mean)/array_std
    return array_nor,array_mean,array_std
    
def invNormMeanStd(array_nor,array_mean,array_std):
    array = array_nor*array_std+array_mean
    return array 

def NormMaxMin(array,dim=0):
    array_max = np.nanmax(array,dim)
    array_min = np.nanmin(array,dim)
    array_nor = (array-array_min)/(array_max-array_min)
    return array_nor,array_max,array_min

def invNormMaxMin(array_nor,array_max,array_min):
    array = array_nor*(array_max-array_min)+array_min
    return array

# miss data
def MissData(array):
    array[np.where(np.isnan(array))] = 0
    return array

# negetive value
def NegeData(array):
    array[np.where(array<0)] = 0
    return array


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # arguments for swin transformer 
    parser.add_argument(
        "--train_input_dir", type=str, help=" the path to So2Sat POP training data",
        default='/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_train')
    parser.add_argument(
        "--test_input_dir", type=str, help=" the path to So2Sat POP testing data",
        default='/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_test')
    parser.add_argument(
        "--model_folder", type=str, help=" the path to save So2Sat POP prediction data",
        default='/p/scratch/deepacf/deeprain/ji4/starter-pack/So2Sat_POP_model')
 
    parser.add_argument(
        "--interval", type=int, help=" split training samples into 'interval' fold",
        default=5)
    parser.add_argument(
        "--train_start_idx_list", type=list, default=[0,4,3],
        help=" using samples strats with train_start_idx for training")
    parser.add_argument(
        "--seed", type=int, default=0,
        help=" seed for reproduceability")   
    
    parser.add_argument(
        "--patch_size", type=tuple, default=(1, 1),
        help=" patch_size in swin transformer")      
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1,
        help=" dropout_rate in swin transformer")       
    parser.add_argument(
        "--num_heads", type=int, default=8,
        help=" number of heads in swin transformer")      
    parser.add_argument(
        "--embed_dim", type=int, default=64,
        help=" number of embed dims in swin transformer")      
    parser.add_argument(
        "--num_mlp", type=int, default=256,
        help=" number of mlp nodes in swin transformer")      
    parser.add_argument(
        "--qkv_bias", type=bool, default=True,
        help=" whether use qkv_bias")       
    parser.add_argument(
        "--window_size", type=int, default=2,
        help=" window_size in swin transformer")      
    parser.add_argument(
        "--shift_size", type=int, default=1,
        help=" shift_size in swin transformer")  

    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help=" learning_rate when training swin transformer") 
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help=" batch_size when training swin transformer")
    parser.add_argument(
        "--num_epochs", type=int, default=80,
        help=" num_epochs when training swin transformer")   
    parser.add_argument(
        "--validation_split", type=float, default=0.1,
        help=" validation_split when training swin transformer")   
    parser.add_argument(
        "--weight_decay", type=float, default=1e-6,
        help=" weight_decay when training swin transformer")   
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help=" label_smoothing when training swin transformer")   
    parser.add_argument(
        "--norm", type=str, default='MaxMin',
        help=" label_smoothing when training swin transformer")      


    # arguments for random forecast
    parser.add_argument(
        "--feature_folder", type=str, help=" the path to So2Sat POP features",
        default='/p/scratch/deepacf/deeprain/ji4/starter-pack/scripts/So2Sat_POP_features')
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
    train_input_dir = args.train_input_dir
    test_input_dir = args.test_input_dir
    model_folder = args.model_folder
    seed = args.seed
    interval = args.interval
    train_start_idx_list = args.train_start_idx_list

    patch_size = args.patch_size
    dropout_rate = args.dropout_rate
    num_heads = args.num_heads    
    embed_dim = args.embed_dim
    num_mlp = args.num_mlp
    qkv_bias = args.qkv_bias        
    window_size = args.window_size
    shift_size = args.shift_size 

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs     
    validation_split = args.validation_split
    weight_decay = args.weight_decay  
    label_smoothing = args.label_smoothing
    norm = args.norm      
   
    feature_folder = args.feature_folder
    category_list = args.category_list
    min_fimportance_list = args.min_fimportance_list
    best_estimator_list = args.best_estimator_list

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    mode = 'classification' # swint for classfication task

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    with xr.open_dataset(os.path.join(train_input_dir,'X_train_sen2_rgb_spring.nc')) as df:
        X_train_sen2_rgb_spring = df['X_train_sen2_rgb_spring']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_sen2_rgb_summer.nc')) as df:
        X_train_sen2_rgb_summer = df['X_train_sen2_rgb_summer']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_sen2_rgb_autumn.nc')) as df:
        X_train_sen2_rgb_autumn = df['X_train_sen2_rgb_autumn']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_sen2_rgb_winter.nc')) as df:
        X_train_sen2_rgb_winter = df['X_train_sen2_rgb_winter']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_viirs.nc')) as df:
        X_train_viirs = df['X_train_viirs']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_lcz.nc')) as df:
        X_train_lcz = df['X_train_lcz']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_dem.nc')) as df:
        X_train_dem = df['X_train_dem']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_lu.nc')) as df:
        X_train_lu = df['X_train_lu']
    with xr.open_dataset(os.path.join(train_input_dir,'X_train_osm.nc')) as df:
        X_train_osm = df['X_train_osm']
        
    with xr.open_dataset(os.path.join(train_input_dir,'y_train_count.nc')) as df:
        Y_train_count = df['y_train_count']
    with xr.open_dataset(os.path.join(train_input_dir,'y_train_class.nc')) as df:
        Y_train_class = df['y_train_class']
    
    train_start_idx = train_start_idx_list[0]
    x_train = np.concatenate((X_train_sen2_rgb_spring.loc[train_start_idx::interval,:,:,:], 
                              X_train_sen2_rgb_summer.loc[train_start_idx::interval,:,:,:], 
                              X_train_sen2_rgb_autumn.loc[train_start_idx::interval,:,:,:], 
                              X_train_sen2_rgb_winter.loc[train_start_idx::interval,:,:,:],
                              X_train_viirs.loc[train_start_idx::interval,:,:,:], 
                              X_train_lcz.loc[train_start_idx::interval,:,:,:],
                              X_train_lu.loc[train_start_idx::interval,:,:,:], 
                              X_train_dem.loc[train_start_idx::interval,:,:,:]), axis=-1)
    x_train_osm = X_train_osm.loc[train_start_idx::interval]
    y_train_count = Y_train_count.loc[train_start_idx::interval]
    y_train_class = Y_train_class.loc[train_start_idx::interval]
    
    image_shape = x_train.shape[1:3]    
    input1_shape = x_train.shape[1:]
    input2_shape = x_train_osm.shape[1:2]    

    for index in range(1, len(train_start_idx_list)):
        train_start_idx = train_start_idx_list[index]
        x_train_temp = np.concatenate(
                         (X_train_sen2_rgb_spring.loc[train_start_idx::interval,:,:,:], 
                          X_train_sen2_rgb_summer.loc[train_start_idx::interval,:,:,:], 
                          X_train_sen2_rgb_autumn.loc[train_start_idx::interval,:,:,:], 
                          X_train_sen2_rgb_winter.loc[train_start_idx::interval,:,:,:],
                          X_train_viirs.loc[train_start_idx::interval,:,:,:], 
                          X_train_lcz.loc[train_start_idx::interval,:,:,:],
                          X_train_lu.loc[train_start_idx::interval,:,:,:], 
                          X_train_dem.loc[train_start_idx::interval,:,:,:]), axis=-1)    
        x_train = np.concatenate((x_train,x_train_temp),axis=0)

        x_train_osm_temp = X_train_osm.loc[train_start_idx::interval] 
        x_train_osm = np.concatenate((x_train_osm,x_train_osm_temp),axis=0)

        y_train_count_temp = Y_train_count.loc[train_start_idx::interval] 
        y_train_count = np.concatenate((y_train_count,y_train_count_temp),axis=0)    

        y_train_class_temp = Y_train_class.loc[train_start_idx::interval] 
        y_train_class = np.concatenate((y_train_class,y_train_class_temp),axis=0)

        del x_train_temp, x_train_osm_temp
        del y_train_count_temp, y_train_class_temp
        gc.collect()

    num_classes = len(np.unique(y_train_class))
    if num_classes == 17: # the training samples have all POP classes
        y_train_class_onehot = keras.utils.to_categorical(y_train_class, num_classes)
        print('>>>>> fatch {} training samples'.format(y_train_class.shape[0]))
    else:
        print("you should fatch more data as training samples!")
        
    if mode=='classification':
        y_train = y_train_class_onehot
    elif mode=='regression':
        y_train = np.reshape(y_train_count,[-1,1])
        
    # building swin transformer 
    image_dimension = image_shape[0]  # Initial image size
    num_patch_x = image_shape[0] // patch_size[0]
    num_patch_y = image_shape[1] // patch_size[1]

    input1 = layers.Input(input1_shape)
    input2 = layers.Input(input2_shape)

    x2 = layers.Dense(1024, activation="relu")(input2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Dense(image_dimension*image_dimension, activation="relu")(x2)
    x2 = layers.Reshape((image_dimension, image_dimension, 1))(x2)

    input = tf.keras.layers.Concatenate(axis=-1)([input1, x2])

    x = layers.experimental.preprocessing.RandomCrop(image_dimension, image_dimension)(input)
    x = layers.experimental.preprocessing.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)

    if mode=='classification':
        output = layers.Dense(num_classes, activation="softmax")(x)
    elif mode=='regression':
        output = layers.Dense(1, activation="linear")(x)
    model = keras.Model([input1,input2], output) 
    model.summary()
    
    if norm == 'MaxMin':
        x_train_nor,x_train_max,x_train_min = NormMaxMin(x_train)
        del x_train; gc.collect()
        x_train_osm_nor,x_train_osm_max,x_train_osm_min = NormMaxMin(x_train_osm)
        del x_train_osm; gc.collect()
        x_train_nor = MissData(x_train_nor)
        x_train_osm_nor = MissData(x_train_osm_nor)
    elif norm == 'MeanStd':
        x_train_nor,x_train_mean,x_train_std = NormMeanStd(x_train)
        del x_train; gc.collect()
        x_train_osm_nor,x_train_osm_mean,x_train_osm_std = NormMeanStd(x_train_osm)
        del x_train_osm; gc.collect()   
        x_train_nor = MissData(x_train_nor)
        x_train_osm_nor = MissData(x_train_osm_nor)

    print("Starting training...\n")
    if mode=='classification':
        y_train_nor = y_train
    elif mode=='regression':
        if norm == 'MaxMin':
            y_train_nor,y_train_max,y_train_min = NormMaxMin(y_train)
            y_train_nor = MissData(y_train_nor)
        elif norm == 'MeanStd':
            y_train_nor,y_train_mean,y_train_std = NormMeanStd(y_train)
            y_train_nor = MissData(y_train_nor)        
          
    if mode=='regression':
        model.compile(
            loss=keras.losses.mse,
            optimizer=tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            metrics=[
                keras.metrics.mse,
                keras.metrics.mae,
            ],
        )
        
    elif mode=='classification':
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

    history = model.fit(
        [x_train_nor,x_train_osm_nor],
        y_train_nor,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=validation_split,
    )
    del x_train_nor,x_train_osm_nor
    gc.collect()
    
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

    # saving models
    # save_model = os.path.join(save_dir,'cross3_batch32_epoch80.h5')
    # history.model.save(save_model)
    save_fig = os.path.join(model_folder,'swin_training_loss')
    plt.savefig(save_fig)
    plt.close()
    
    print("Starting predicting POP class using swint...\n")
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_sen2_rgb_spring.nc')) as df:
        X_train_sen2_rgb_spring = df['X_train_sen2_rgb_spring']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_sen2_rgb_summer.nc')) as df:
        X_train_sen2_rgb_summer = df['X_train_sen2_rgb_summer']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_sen2_rgb_autumn.nc')) as df:
        X_train_sen2_rgb_autumn = df['X_train_sen2_rgb_autumn']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_sen2_rgb_winter.nc')) as df:
        X_train_sen2_rgb_winter = df['X_train_sen2_rgb_winter']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_viirs.nc')) as df:
        X_train_viirs = df['X_train_viirs']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_lcz.nc')) as df:
        X_train_lcz = df['X_train_lcz']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_dem.nc')) as df:
        X_train_dem = df['X_train_dem']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_lu.nc')) as df:
        X_train_lu = df['X_train_lu']
    with xr.open_dataset(os.path.join(test_input_dir,'X_train_osm.nc')) as df:
        X_train_osm = df['X_train_osm']
        valid_city_name = df['city_name']
        valid_grid_idx = df['grid_idx']
        
    x_valid = np.concatenate((X_train_sen2_rgb_spring,
                          X_train_sen2_rgb_summer, 
                          X_train_sen2_rgb_autumn, 
                          X_train_sen2_rgb_winter,
                          X_train_viirs, 
                          X_train_lcz,
                          X_train_lu, 
                          X_train_dem), axis=-1)   
    x_valid_osm = X_train_osm
    
    if norm == 'MaxMin':    
        x_valid_nor = (x_valid-x_train_min)/(x_train_max-x_train_min)
        x_valid_osm_nor = (x_valid_osm-x_train_osm_min)/(x_train_osm_max-x_train_osm_min)
        del x_valid,x_valid_osm
        gc.collect()

    elif norm == 'MeanStd':
        x_valid_nor = (x_valid-x_train_mean)/x_train_std
        x_valid_osm_nor = (x_valid_osm-x_train_osm_mean)/x_train_osm_std
        del x_valid,x_valid_osm
        gc.collect()

    x_valid_nor = np.nan_to_num(x_valid_nor)
    x_valid_osm_nor = np.nan_to_num(x_valid_osm_nor)        

    if mode=='classification':
        f_valid_onehot = model.predict([x_valid_nor,x_valid_osm_nor])
        f_valid = f_valid_onehot.argmax(axis=-1)
    elif mode=='regression':
        f_valid_nor = model.predict([x_valid_nor,x_valid_osm_nor])
        if norm == 'MaxMin':   
            f_valid = invNormMaxMin(f_valid_nor,y_train_max,y_train_min)
        elif norm == 'MeanStd':
            f_valid = invNormStd(f_valid_nor,y_train_mean,y_train_std)
        f_valid = NegeData(f_valid)

    if mode=='regression':
        pred_csv_path = os.path.join(model_folder,'swint_reg_predictions.csv')
    elif mode=='classification':
        pred_csv_path = os.path.join(model_folder,'swint_cla_predictions.csv')

    df_pred = pd.DataFrame()
    df_pred["CITY"] = valid_city_name
    df_pred["GRD_ID"] = valid_grid_idx
    df_pred['Predictions'] = f_valid.ravel()
    df_pred.to_csv(pred_csv_path, index=False)

    print(">>>>>>>>>> swint predict POP class in: {}".format(pred_csv_path))


    print("Starting regression using random forest")
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

    # merge predicted POP class into testing dataframe
    pred_class_df = pd.read_csv(pred_csv_path)      
    test_df = test_df.merge(pred_class_df, 
            on=['CITY', 'GRD_ID'], how='outer', suffixes=['_ref', '_pred'])
    y_test_label = test_df[pred_label_col_reg]

    # prepare the prediction dataframe
    df_pred_complete = pd.DataFrame()

    # bug here
    for icate in range(len(category_list)-2):
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

        # Feature importances
        print("Creation of feature importance plot...\n")
        importances = best_estimator.feature_importances_  # Save feature importances from the model
        path_plot = os.path.join(model_folder, "rf_feature_importance_cate_%s" % str(icate))  # path to saved plot
        plot_feature_importance(importances, x_test, path_plot)

    pred_csv_path = os.path.join(model_folder, 'rf_reg_predictions.csv')
    df_pred_complete.to_csv(pred_csv_path, index=False)
    # return pred_csv_path


