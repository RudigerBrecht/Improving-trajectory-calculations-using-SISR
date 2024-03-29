import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

import time

###

def mynormalize(x):
    return 2*(x-np.min(x))/(np.max(x)-np.min(x))-1
###

udata=np.zeros((137,360,720,75*3),dtype=np.float32)

ds = nc.Dataset('../download_ear5_data/data_train.nc')
u1=ds['u'][0:137,0:360,0:720,0:3*75]

for i in range(137):
    for j in range(3*75):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:3*75]=u1

np.save('normalized_train_u_0_50',udata[0:50,])
np.save('normalized_train_u_50_137',udata[50:137,])

###


udata=np.zeros((137,360,720,75*3),dtype=np.float32)

ds = nc.Dataset('../download_ear5_data/data_train.nc')
u1=ds['v'][0:137,0:360,0:720,0:3*75]

for i in range(137):
    for j in range(3*75):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:3*75]=u1

np.save('normalized_train_v_0_50',udata[0:50,])
np.save('normalized_train_v_50_137',udata[50:137,])


###

ds = nc.Dataset('../download_ear5_data/data_train.nc')
u1=ds['u'][0:137,0:360,0:720,226:236]
for i in range(137):
    for j in range(10):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])
        
np.save('normalized_val_u',udata)

###

ds = nc.Dataset('../download_ear5_data/data_train.nc')
u1=ds['v'][0:137,0:360,0:720,226:236]
for i in range(137):
    for j in range(10):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])
        
np.save('normalized_val_v',udata)
