import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import cv2

import tensorflow as tf

import time

###


def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

###

def scaleup_50(model,field,ds):
    
    
    u_nn=np.zeros((50, 360, 720, 4*24))
    tmp=ds[field][0:50,0:360:2,0:720:2,0:4*24]
    
    start_time = time.time()

    for i in range(50):
        for j in range(4*24):
            uin=tmp[i,:,:,j]
            mi=uin.min()
            ma=uin.max()
            u_nn[i,:,:,j]=interval_mapping(model.predict(interval_mapping(uin[None,],mi,ma,-1,1))[0,:,:,0],-1,1,mi,ma)
            
    print("--- %s seconds --- for 50" % (time.time() - start_time))

    ds[field][0:50,0:360,0:720,0:4*24]=u_nn
    


    
    
    
def scaleup_137(model,field,ds):
    

    
    u_nn=np.zeros((87, 360, 720, 4*24))
    tmp=ds[field][50:137,0:360:2,0:720:2,0:4*24]
    
    start_time = time.time()
    
    for i in range(87):
        for j in range(4*24):
            uin=tmp[i,:,:,j]
            mi=uin.min()
            ma=uin.max()
            u_nn[i,:,:,j]=interval_mapping(model.predict(interval_mapping(uin[None,],mi,ma,-1,1))[0,:,:,0],-1,1,mi,ma)

    print("--- %s seconds --- for 137" % (time.time() - start_time))

    ds[field][50:137,0:360,0:720,0:4*24]=u_nn
    


###
ds_nn_u = nc.Dataset('nn_eval_u_m4.nc',"r+")
ds_nn_v = nc.Dataset('nn_eval_v_m4.nc',"r+")


for k in range(4):
    if(k==0):
        model = tf.keras.models.load_model('../trained_nn_models/model_4_u_lvl050') 
        print('scale up u 50')
        scaleup_50(model,'u', ds_nn_u)
        
    if(k==1):
        model = tf.keras.models.load_model('../trained_nn_models/model_4_u_lvl50137')
        print('scale up u 137')
        scaleup_137(model,'u', ds_nn_u)
        
    if(k==2):
        model = tf.keras.models.load_model('../trained_nn_models/model_4_v_lvl050')
        print('scale up v 50')
        scaleup_50(model,'v', ds_nn_v)
        
    if(k==3):
        model = tf.keras.models.load_model('../trained_nn_models/model_4_v_lvl50137')
        print('scale up v 137')
        scaleup_137(model,'v', ds_nn_v)
        

ds_nn_u.close()
ds_nn_v.close()