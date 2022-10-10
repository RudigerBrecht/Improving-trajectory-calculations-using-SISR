import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import cv2

import tensorflow as tf

import time

from scipy import interpolate

###

filenames=['lin_data_eval_jan.nc','lin_data_eval_apr.nc','lin_data_eval_jul.nc','lin_data_eval_oct']

x = np.arange(0,360, 2)
y = np.arange(0,720, 2)

xnew = np.arange(0,360, 1)
ynew = np.arange(0,720, 1)

ulin=np.zeros((4,360,720))
vlin=np.zeros((4,360,720))

###
def scaleup(field):
    
    
    u_lin=np.zeros((137, 360, 720, 74))
    tmp=ds_lin[field][0:137,0:360:2,0:720:2,0:74]
    
    save_field=np.zeros((360,720))
    
    start_time = time.time()

    for i in range(137):
        for j in range(74):
            
            fu = interpolate.interp2d(x, y, tmp[i,:,:,j].transpose(), kind='linear')
            
            u_lin[i,:,:,j] = fu(xnew, ynew).transpose()
            
            if i==10 and j==0:
                save_field=u_lin[i,:,:,j]
                

    print("--- %s seconds --- for linear" % (time.time() - start_time))
    
    ds_lin[field][0:137,0:360,0:720,0:74]=u_lin
    
    return save_field





###
for n in range(4):
    
    ds_lin = nc.Dataset('lin_'+filenames[n],"r+")
    

    for k in range(2):
        if(k==0):
            print('scale up u')
            ulin[n,]=scaleup('u')
            
        if(k==1):
            print('scale up v')
            vlin[n,]=scaleup('v')


    ds_lin.close()
    
np.save('../data1/ulin.npy','ulin')
np.save('../data1/vlin.npy','vlin')

###
###

x = np.arange(0,360, 4)
y = np.arange(0,720, 4)

xnew = np.arange(0,360, 1)
ynew = np.arange(0,720, 1)

ulin=np.zeros((4,360,720))
vlin=np.zeros((4,360,720))

###
def scaleup(field):
    
    
    u_lin=np.zeros((137, 360, 720, 74))
    tmp=ds_lin[field][0:137,0:360:4,0:720:4,0:74]
    
    save_field=np.zeros((360,720))
    
    start_time = time.time()

    for i in range(137):
        for j in range(74):
            
            fu = interpolate.interp2d(x, y, tmp[i,:,:,j].transpose(), kind='linear')
            
            u_lin[i,:,:,j] = fu(xnew, ynew).transpose()
            
            if i==10 and j==0:
                save_field=u_lin[i,:,:,j]
                

    print("--- %s seconds --- for linear" % (time.time() - start_time))
    
    ds_lin[field][0:137,0:360,0:720,0:74]=u_lin
    
    return save_field





###
for n in range(4):
    
    ds_lin = nc.Dataset('lin_2_'+filenames[n],"r+")
    

    for k in range(2):
        if(k==0):
            print('scale up u')
            ulin[n,]=scaleup('u')
            
        if(k==1):
            print('scale up v')
            vlin[n,]=scaleup('v')


    ds_lin.close()
    
np.save('../data2/ulin.npy','ulin')
np.save('../data2/vlin.npy','vlin')

