import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d

from datetime import datetime, timedelta
from netCDF4 import Dataset

from scipy import interpolate
import cv2
from skimage.metrics import structural_similarity as ssim

import tensorflow as tf

import time

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, \
  Activation, Dropout, Concatenate, BatchNormalization, Add, Lambda, DepthwiseConv2D

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras import metrics, losses


###

def mynormalize(x):
    return 2.0*(x-np.min(x))/(np.max(x)-np.min(x))-1.0

###

class CAmodule(tf.keras.layers.Layer):
    def __init__(self, numFilter ):
        super(CAmodul,self).__init__()
        self.gobal  = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape= tf.keras.layers.Reshape((1,-1))
        self.conv1  = tf.keras.layers.Conv1D(numFilter,3,padding='same')
        self.ReLu   = tf.keras.layers.LeakyReLU()#('relu')
        self.conv2  = tf.keras.layers.Conv1D(numFilter,3,padding='same')
        self.Sigmoid= tf.keras.layers.Activation('sigmoid')
        self.mult   = tf.keras.layers.Multiply()
    #@tf.function    
    def call(self,inputs):
        x = self.gobal(inputs)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.ReLu(x)
        x = self.conv2(x)
        x = self.Sigmoid(x)
        x = self.mult([inputs,x])
        
        return x

# Upsample architecture
def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """Creates an EDSR model."""
    x_in = Input(shape=(None, None, 1))

    x = b = Conv2D(num_filters, 3, padding='same')(x_in) #3
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
        
    b = Conv2D(num_filters, 3, padding='same')(b) #3
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    
    x = Conv2D(1, 3, padding='same', activation='linear')(x) # tanh # linear
    

    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    """Creates an EDSR residual block."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in) #3
    x = Conv2D(filters, 3, padding='same')(x) # 3
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
        x = CAmodule(x.shape[-1])(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x) #3
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
    elif scale == 6:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
        x = upsample_1(x, 2, name='conv2d_3_scale_2')

    return x

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


###

model = edsr(scale=2, num_res_blocks=8)
model.summary()

optim_edsr = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))

#model.compile(optimizer=optim_edsr, loss='mean_absolute_error')
model.compile(optimizer='adam', loss='mean_absolute_error')

###
print('load data')
##
u1 = np.load('../preprocess_data/normalized_train_u_50_137.npy')

u_4=np.reshape(np.transpose(u1[:,0:360:2,0:720:2,:], axes=[0, 3, 1,2]),(300*87,180,360,1))
u_2=np.reshape(np.transpose(u1[:,0:360,0:720,:], axes=[0, 3, 1,2]),(300*87,360,720,1))

ds_train = tf.data.Dataset.from_tensor_slices((u_4,u_2))
ds_train = ds_train.shuffle(len(u_4)).batch(16)
ds_train = ds_train.cache().prefetch(tf.data.experimental.AUTOTUNE)

#

u1 = np.load('../preprocess_data/normalized_val_u.npy')[50:137,]
        
u_2_val=np.reshape(np.transpose(u1[:,0:360:2,0:720:2,:], axes=[0, 3, 1,2]),(87*10,180,360,1))
u_1_val=np.reshape(np.transpose(u1[:,:,:,:], axes=[0, 3, 1,2]),(87*10,360,720,1))

ds_val = tf.data.Dataset.from_tensor_slices((u_2_val,u_1_val))
ds_val = ds_val.shuffle(len(u_2[::500,])).batch(5)
ds_val = ds_val.cache().prefetch(tf.data.experimental.AUTOTUNE)

##
print('start training')

model.fit(ds_train, epochs=25, validation_data=ds_val, callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose = 1)])

##
model.save('../trained_nn_models/model_1_u_lvl50137')
