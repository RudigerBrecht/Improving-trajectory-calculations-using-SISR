import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

import time

###

def mynormalize(x):
    return 2*(x-np.min(x))/(np.max(x)-np.min(x))-1
###

udata=np.zeros((87,360,720,75*4),dtype=np.float32)

ds = nc.Dataset('../download_era5_data/train_u.nc')
u1=ds['u'][50:137,0:360,0:720,0:75*4]

for i in range(87):
    for j in range(75):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:75]=u1

np.save('normalized_train_u_50_137',udata)

###


udata=np.zeros((50,360,720,75*4),dtype=np.float32)

ds = nc.Dataset('../download_era5_data/train_u.nc')
u1=ds['u'][0:50,0:360,0:720,0:75*4]

for i in range(50):
    for j in range(75):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:75]=u1

np.save('normalized_train_u_0_50',udata)

###

udata=np.zeros((137,360,720,10),dtype=np.float32)

ds = nc.Dataset('../download_era5_data/eval_u.nc')
u1=ds['u'][0:137,0:360,0:720,0:10]

for i in range(137):
    for j in range(10):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:75]=u1

np.save('normalized_val_u',udata)

###

vdata=np.zeros((87,360,720,75*4),dtype=np.float32)

ds = nc.Dataset('../download_era5_data/train_v.nc')
u1=ds['v'][50:137,0:360,0:720,0:75*4]

for i in range(87):
    for j in range(75):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:75]=u1

np.save('normalized_train_v_50_137',udata)


###

vdata=np.zeros((50,360,720,75*4),dtype=np.float32)

ds = nc.Dataset('../download_era5_data/train_v.nc')
u1=ds['v'][0:50,0:360,0:720,0:75*4]

for i in range(50):
    for j in range(75):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:75]=u1

np.save('normalized_train_v_0_50',udata)

###

udata=np.zeros((137,360,720,10),dtype=np.float32)

ds = nc.Dataset('../download_era5_data/eval_v.nc')
u1=ds['v'][0:137,0:360,0:720,0:10]

for i in range(137):
    for j in range(10):
        u1[i,:,:,j]=mynormalize(u1[i,:,:,j])

udata[:,:,:,0:75]=u1

np.save('normalized_val_v',udata)
