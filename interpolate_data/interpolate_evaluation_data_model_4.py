import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import cv2

import tensorflow as tf

import time

###

unn=np.zeros((4,360,720))
vnn=np.zeros((4,360,720))
uref=np.zeros((4,360,720))
vref=np.zeros((4,360,720))


filenames=['data_eval_jan.nc','data_eval_apr.nc','data_eval_jul.nc','data_eval_oct']

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

###

def scaleup_50(model,field):
    
    
    u_nn=np.zeros((50, 360, 720, 74))
    tmp=ds_nn[field][0:50,0:360:4,0:720:4,0:74]
    
    save_field=np.zeros((360,720))
    save_field_ref=np.zeros((360,720))
    
    start_time = time.time()

    for i in range(50):
        for j in range(74):
            uin=tmp[i,:,:,j]
            mi=uin.min()
            ma=uin.max()
            u_nn[i,:,:,j]=interval_mapping(model.predict(model.predict(interval_mapping(uin[None,],mi,ma,-1,1)))[0,:,:,0],-1,1,mi,ma)
            
            if i==10 and j==0:
                save_field=u_nn[i,:,:,j]
                save_field_ref=uin
            
    print("--- %s seconds --- for 50" % (time.time() - start_time))

    ds_nn[field][0:50,0:360,0:720,0:74]=u_nn
    
    return save_field, save_field_ref
    


    
    
    
def scaleup_137(model,field):
    

    
    u_nn=np.zeros((87, 360, 720, 74))
    tmp=ds_nn[field][50:137,0:360:4,0:720:4,0:74]
    
    start_time = time.time()
    
    for i in range(87):
        for j in range(74):
            uin=tmp[i,:,:,j]
            mi=uin.min()
            ma=uin.max()
            u_nn[i,:,:,j]=interval_mapping(model.predict(model.predict(interval_mapping(uin[None,],mi,ma,-1,1)))[0,:,:,0],-1,1,mi,ma)

    print("--- %s seconds --- for 137" % (time.time() - start_time))

    ds_nn[field][50:137,0:360,0:720,0:74]=u_nn
    


###
for n in range(3):
    ds_nn = nc.Dataset('nn2_'+filenames[n],"r+")
    
    

    for k in range(4):
        if(k==0):
            model = tf.keras.models.load_model('../trained_nn_models/model_4_u_lvl050')
            print('scale up u 50')
            unn[n,],uref[n,]=scaleup_50(model,'u')
            
        if(k==1):
            model = tf.keras.models.load_model('../trained_nn_models/model_4_u_lvl50137')
            print('scale up u 137')
            scaleup_137(model,'u')
            
        if(k==2):
            model = tf.keras.models.load_model('../trained_nn_models/model_4_v_lvl050')
            print('scale up v 50')
            vnn[n,],vref[n,]=scaleup_50(model,'v')
            
        if(k==3):
            model = tf.keras.models.load_model('../trained_nn_models/model_4_v_lvl50137')
            print('scale up v 137')
            scaleup_137(model,'v')
            

    ds_nn.close()
    
np.save('../data2/unn.npy','unn')
np.save('../data2/vnn.npy','vnn')
np.save('../data2/uref.npy','uref')
np.save('../data2/vref.npy','vref')
