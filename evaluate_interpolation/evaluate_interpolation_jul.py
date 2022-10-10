import netCDF4 as nc

import numpy as np
from netCDF4 import Dataset
from scipy import interpolate
###

def mynormalize(x):
    return 2.0*(x-np.min(x))/(np.max(x)-np.min(x))-1.0

def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def rmse(xp, xt):
    return np.sqrt(((xp - xt)**2).mean())

# def rmse(xp,yp, xt,yt):
#     return np.sqrt(((xp - xt)**2+(yp - yt)**2).mean())

def lininter(field,fin,fout):
    x = np.arange(0,360, fin)
    y = np.arange(0,720, fin)
    fu = interpolate.interp2d(x, y, field.transpose(), kind='linear')
    xnew = np.arange(0,360, fout)
    ynew = np.arange(0,720, fout)
    return fu(xnew, ynew).transpose()

def modelinter(model,field):
    mi=field.min()
    ma=field.max()
    return interval_mapping(model.predict(interval_mapping(field[None,],mi,ma,-1,1))[0,:,:,0],-1,1,mi,ma)



modelu1_050 = tf.keras.models.load_model('../trained_nn_models/model_1_u_lvl050')
modelu1_50137 = tf.keras.models.load_model('../trained_nn_models/model_1_u_lvl50137')
modelv1_050 = tf.keras.models.load_model('../trained_nn_models/model_1_v_lvl050')
modelv1_50137 = tf.keras.models.load_model('../trained_nn_models/model_1_v_lvl50137')

modelu2_050 = tf.keras.models.load_model('../trained_nn_models/model_2_u_lvl050')
modelu2_50137 = tf.keras.models.load_model('../trained_nn_models/model_2_u_lvl50137')
modelv2_050 = tf.keras.models.load_model('../trained_nn_models/model_2_v_lvl050')
modelv2_50137 = tf.keras.models.load_model('../trained_nn_models/model_2_v_lvl50137')

modelu4_050 = tf.keras.models.load_model('../trained_nn_models/model_4_u_lvl050')
modelu4_50137 = tf.keras.models.load_model('../trained_nn_models/model_4_u_lvl50137')
modelv4_050 = tf.keras.models.load_model('../trained_nn_models/model_4_v_lvl050')
modelv4_50137 = tf.keras.models.load_model('../trained_nn_models/model_4_v_lvl50137')

###
ds_ref = nc.Dataset('download_ear5_data/data_eval_jul.nc')
uref=ds_ref['u'][0:136, 0:360,0:720,0:24]
vref=ds_ref['v'][0:136, 0:360,0:720,0:24]

###
rmse_u_lin=[0,0,0]
rmse_u_nn4=[0,0,0]
rmse_u_nn2=[0,0]
rmse_u_nn1=0

rmse_v_lin=[0,0,0]
rmse_v_nn4=[0,0,0]
rmse_v_nn2=[0,0]
rmse_v_nn1=0

ssim_u_lin=[0,0,0]
ssim_u_nn4=[0,0,0]
ssim_u_nn2=[0,0]
ssim_u_nn1=0

ssim_v_lin=[0,0,0]
ssim_v_nn4=[0,0,0]
ssim_v_nn2=[0,0]
ssim_v_nn1=0
###


for i in range(136):
    for t in range(24):
        
        ulin4_4=lininter(uref[i,0:360:8,0:720:8,t],8,4)
        ulin4_2=lininter(uref[i,0:360:4,0:720:4,t],4,2)
        ulin4_1=lininter(uref[i,0:360:2,0:720:2,t],2,1)
        vlin4_4=lininter(vref[i,0:360:8,0:720:8,t],8,4)
        vlin4_2=lininter(vref[i,0:360:4,0:720:4,t],4,2)
        vlin4_1=lininter(vref[i,0:360:2,0:720:2,t],2,1)
        
        rmse_u_lin[0]+=rmse(ulin4_4,uref[i,0:360:4,0:720:4,t])/136/24
        rmse_u_lin[1]+=rmse(ulin4_2,uref[i,0:360:2,0:720:2,t])/136/24
        rmse_u_lin[2]+=rmse(ulin4_1,uref[i,0:360,0:720,t])/136/24
        
        rmse_v_lin[0]+=rmse(vlin4_4,vref[i,0:360:4,0:720:4,t])/136/24
        rmse_v_lin[1]+=rmse(vlin4_2,vref[i,0:360:2,0:720:2,t])/136/24
        rmse_v_lin[2]+=rmse(vlin4_1,vref[i,0:360,0:720,t])/136/24
        
        ##
        if i <=50:
            unn4_4=modelinter(modelu4_050,uref[i,0:360:8,0:720:8,t])
            unn4_2=modelinter(modelu4_050,uref[i,0:360:4,0:720:4,t])
            unn4_1=modelinter(modelu4_050,uref[i,0:360:2,0:720:2,t])
            vnn4_4=modelinter(modelv4_050,vref[i,0:360:8,0:720:8,t])
            vnn4_2=modelinter(modelv4_050,vref[i,0:360:4,0:720:4,t])
            vnn4_1=modelinter(modelv4_050,vref[i,0:360:2,0:720:2,t])
        else:
            unn4_4=modelinter(modelu4_50137,uref[i,0:360:8,0:720:8,t])
            unn4_2=modelinter(modelu4_50137,uref[i,0:360:4,0:720:4,t])
            unn4_1=modelinter(modelu4_50137,uref[i,0:360:2,0:720:2,t])
            vnn4_4=modelinter(modelv4_50137,vref[i,0:360:8,0:720:8,t])
            vnn4_2=modelinter(modelv4_50137,vref[i,0:360:4,0:720:4,t])
            vnn4_1=modelinter(modelv4_50137,vref[i,0:360:2,0:720:2,t])
        
        rmse_u_nn4[0]+=rmse(unn4_4,uref[i,0:360:4,0:720:4,t])/136/24
        rmse_u_nn4[1]+=rmse(unn4_2,uref[i,0:360:2,0:720:2,t])/136/24
        rmse_u_nn4[2]+=rmse(unn4_1,uref[i,0:360,0:720,t])/136/24
        
        rmse_v_nn4[0]+=rmse(vnn4_4,vref[i,0:360:4,0:720:4,t])/136/24
        rmse_v_nn4[1]+=rmse(vnn4_2,vref[i,0:360:2,0:720:2,t])/136/24
        rmse_v_nn4[2]+=rmse(vnn4_1,vref[i,0:360,0:720,t])/136/24
        
        ##
        if i <= 50:
            unn2_2=modelinter(modelu2_050,uref[i,0:360:4,0:720:4,t])
            unn2_1=modelinter(modelu2_050,uref[i,0:360:2,0:720:2,t])
            vnn2_2=modelinter(modelv2_050,vref[i,0:360:4,0:720:4,t])
            vnn2_1=modelinter(modelv2_050,vref[i,0:360:2,0:720:2,t])        
        else:
            unn2_2=modelinter(modelu2_50137,uref[i,0:360:4,0:720:4,t])
            unn2_1=modelinter(modelu2_50137,uref[i,0:360:2,0:720:2,t])
            vnn2_2=modelinter(modelv2_50137,vref[i,0:360:4,0:720:4,t])
            vnn2_1=modelinter(modelv2_50137,vref[i,0:360:2,0:720:2,t])

        rmse_u_nn2[0]+=rmse(unn2_2,uref[i,0:360:2,0:720:2,t])/136/24
        rmse_u_nn2[1]+=rmse(unn2_1,uref[i,0:360,0:720,t])/136/24
        
        rmse_v_nn2[0]+=rmse(vnn2_2,vref[i,0:360:2,0:720:2,t])/136/24
        rmse_v_nn2[1]+=rmse(vnn2_1,vref[i,0:360,0:720,t])/136/24
        
        ##
        if i <= 50:
            unn1_1=modelinter(modelu1_050,uref[i,0:360:2,0:720:2,t])
            vnn1_1=modelinter(modelv1_050,vref[i,0:360:2,0:720:2,t])
        else:
            unn1_1=modelinter(modelu1_50137,uref[i,0:360:2,0:720:2,t])
            vnn1_1=modelinter(modelv1_50137,vref[i,0:360:2,0:720:2,t])
        
        rmse_u_nn1+=rmse(unn1_1,uref[i,0:360,0:720,t])/136/24
        rmse_v_nn1+=rmse(vnn1_1,vref[i,0:360,0:720,t])/136/24

        
    # ssim 
    
        ssim_u_lin[0]+=ssim(ulin4_4,uref[i,0:360:4,0:720:4,t])/136/24
        ssim_u_lin[1]+=ssim(ulin4_2,uref[i,0:360:2,0:720:2,t])/136/24
        ssim_u_lin[2]+=ssim(ulin4_1,uref[i,0:360,0:720,t])/136/24
        
        ssim_u_nn4[0]+=ssim(unn4_4,uref[i,0:360:4,0:720:4,t])/136/24
        ssim_u_nn4[1]+=ssim(unn4_2,uref[i,0:360:2,0:720:2,t])/136/24
        ssim_u_nn4[2]+=ssim(unn4_1,uref[i,0:360,0:720,t])/136/24
        #
        ssim_u_nn2[0]+=ssim(unn2_2,uref[i,0:360:2,0:720:2,t])/136/24
        ssim_u_nn2[1]+=ssim(unn2_1,uref[i,0:360,0:720,t])/136/24
        #
        ssim_u_nn1+=ssim(unn1_1,uref[i,0:360,0:720,t])/136/24

    
        ssim_v_lin[0]+=ssim(vlin4_4,vref[i,0:360:4,0:720:4,t])/136/24
        ssim_v_lin[1]+=ssim(vlin4_2,vref[i,0:360:2,0:720:2,t])/136/24
        ssim_v_lin[2]+=ssim(vlin4_1,vref[i,0:360,0:720,t])/136/24
        
        ssim_v_nn4[0]+=ssim(vnn4_4,vref[i,0:360:4,0:720:4,t])/136/24
        ssim_v_nn4[1]+=ssim(vnn4_2,vref[i,0:360:2,0:720:2,t])/136/24
        ssim_v_nn4[2]+=ssim(vnn4_1,vref[i,0:360,0:720,t])/136/24
        #
        ssim_v_nn2[0]+=ssim(vnn2_2,vref[i,0:360:2,0:720:2,t])/136/24
        ssim_v_nn2[1]+=ssim(vnn2_1,vref[i,0:360,0:720,t])/136/24
        #
        ssim_v_nn1+=ssim(vnn1_1,vref[i,0:360,0:720,t])/136/24




###

np.save('../errdata/rmse_u_lin_jul',rmse_u_lin)
np.save('../errdata/rmse_u_nn4_jul',rmse_u_nn4)
np.save('../errdata/rmse_u_nn2_jul',rmse_u_nn2)
np.save('../errdata/rmse_u_nn1_jul',rmse_u_nn1)

np.save('../errdata/rmse_v_lin_jul',rmse_v_lin)
np.save('../errdata/rmse_v_nn4_jul',rmse_v_nn4)
np.save('../errdata/rmse_v_nn2_jul',rmse_v_nn2)
np.save('../errdata/rmse_v_nn1_jul',rmse_v_nn1)

np.save('../errdata/ssim_u_lin_jul',ssim_u_lin)
np.save('../errdata/ssim_u_nn4_jul',ssim_u_nn4)
np.save('../errdata/ssim_u_nn2_jul',ssim_u_nn2)
np.save('../errdata/ssim_u_nn1_jul',ssim_u_nn1)

np.save('../errdata/ssim_v_lin_jul',ssim_v_lin)
np.save('../errdata/ssim_v_nn4_jul',ssim_v_nn4)
np.save('../errdata/ssim_v_nn2_jul',ssim_v_nn2)
np.save('../errdata/ssim_v_nn1_jul',ssim_v_nn1)