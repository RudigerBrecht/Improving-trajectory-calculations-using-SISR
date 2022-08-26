import netCDF4 as nc

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
import cv2

import tensorflow as tf

import time

from scipy import interpolate

###

x = np.arange(0,360, 2)
y = np.arange(0,720, 2)

xnew = np.arange(0,360, 1)
ynew = np.arange(0,720, 1)


###
def scaleup(field):


    u_lin=np.zeros((137, 360, 720, 4*24))
    tmp=ds_lin[field][0:137,0:360:2,0:720:2,0:4*24]

    start_time = time.time()

    for i in range(137):
        for j in range(4*24):

            fu = interpolate.interp2d(x, y, tmp[i,:,:,j].transpose(), kind='linear')

            u_lin[i,:,:,j] = fu(xnew, ynew).transpose()

    print("--- %s seconds --- for linear" % (time.time() - start_time))

    ds_lin[field][0:137,0:360,0:720,0:4*24]=u_lin





###
ds_lin_u = nc.Dataset('lin_eval_u.nc',"r+")
ds_lin_v = nc.Dataset('lin_eval_v.nc',"r+")

for k in range(2):
    if(k==0):
        print('scale up u')
        scaleup('u',ds_lin_u)

    if(k==1):
        print('scale up v')
        scaleup('v',ds_lin_v)


ds_lin.close()