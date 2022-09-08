import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import cv2

import tensorflow as tf

import time

###

filenames=['nn_wind_orig_20000113140000','wind_orig_20000413050000.nc','wind_orig_20000712200000.nc','wind_orig_20001014140000.nc']

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

###

def scaleup_50(model,field):
    
    
    u_nn=np.zeros((50, 360, 720, 24))
    tmp=ds_nn[field][0:50,0:360:2,0:720:2,0:24]
    
    start_time = time.time()

    for i in range(50):
        for j in range(24):
            uin=tmp[i,:,:,j]
            mi=uin.min()
            ma=uin.max()
            u_nn[i,:,:,j]=interval_mapping(model.predict(interval_mapping(uin[None,],mi,ma,-1,1))[0,:,:,0],-1,1,mi,ma)
            
    print("--- %s seconds --- for 50" % (time.time() - start_time))

    ds_nn[field][0:50,0:360,0:720,0:24]=u_nn
    


    
    
    
def scaleup_137(model,field):
    

    
    u_nn=np.zeros((87, 360, 720, 24))
    tmp=ds_nn[field][50:137,0:360:2,0:720:2,0:24]
    
    start_time = time.time()
    
    for i in range(87):
        for j in range(24):
            uin=tmp[i,:,:,j]
            mi=uin.min()
            ma=uin.max()
            u_nn[i,:,:,j]=interval_mapping(model.predict(interval_mapping(uin[None,],mi,ma,-1,1))[0,:,:,0],-1,1,mi,ma)

    print("--- %s seconds --- for 137" % (time.time() - start_time))

    ds_nn[field][50:137,0:360,0:720,0:24]=u_nn
    


###
for n in range(3):
    ds_nn = nc.Dataset('nn_'+filenames[n],"r+")
    
    

    for k in range(4):
        if(k==0):
            model = tf.keras.models.load_model('models/u_50') # model u_137 would give better results
            print('scale up u 50')
            scaleup_50(model,'u')
            
        if(k==1):
            model = tf.keras.models.load_model('models/u_137')
            print('scale up u 137')
            scaleup_137(model,'u')
            
        if(k==2):
            model = tf.keras.models.load_model('models/v_50')
            print('scale up v 50')
            scaleup_50(model,'v')
            
        if(k==3):
            model = tf.keras.models.load_model('models/v_137')
            print('scale up v 137')
            scaleup_137(model,'v')
            

    ds_nn.close()
                
            
            
            

    
