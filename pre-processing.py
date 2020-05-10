import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import random as r
import math
import pickle
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import keras.backend.tensorflow_backend as tfback
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


K.set_image_data_format('channels_first')


img_size = 240      #original img size is 240*240
smooth = 0.005 
num_of_aug = 2
num_epoch = 30
pul_seq = 'Flair'
sharp = False       # sharpen filter
LR = 1e-4

num_of_patch = 4 #must be a square number
label_num = 5   # 1 = necrosis+NET, 2 = tumor core,3= original, 4 = ET, 5 = complete tumor
'''
0: other
1: necrosis + NET
2: edema
4: enhancing tumor
5: full tumor
'''

import glob
#function to read one subject data
def create_data_onesubject_val(src, mask,count, label=False):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    k = len(files) - count -1
    imgs = []
    file = files[k]
    print('Processing---', mask,'--',file)
    
    img = io.imread(file, plugin='simpleitk')
    #img = trans.resize(img, resize, mode='constant')
    if label:
        if label_num == 5:
            img[img != 0] = 1       #Region 1 => 1+2+3+4 complete tumor
        if label_num == 1:
            img[img != 1] = 0       #only left necrosis
        if label_num == 2:
            img[img == 2] = 0       #turn edema to 0
            img[img != 0] = 1       #only keep necrosis, ET, NET = Tumor core
        if label_num == 4:
            img[img != 4] = 0       #only left ET
            img[img == 4] = 1
        img = img.astype('float32')
    else:
        img = (img-img.mean()) / img.std()      #normalization => zero mean   !!!care for the std=0 problem
        img = img.astype('float32')
    for slice in range(155):     #choose the slice range
        img_t = img[slice,:,:]
        img_t =img_t.reshape((1,)+img_t.shape)
        img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
        #img_g = augmentation(img_t,num_of_aug)
        for n in range(img_t.shape[0]):
            imgs.append(img_t[n,:,:,:])
    
    return np.array(imgs)

def create_data(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    r.seed(9)
    r.shuffle(files)    # shuffle patients
    imgs = []
    print('Processing---', mask)
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        #img = trans.resize(img, resize, mode='constant')
        if label:
            if label_num == 5:
                img[img != 0] = 1       #Region 1 => 1+2+3+4 complete tumor
            if label_num == 1:
                img[img != 1] = 0       #only left necrosis and NET
            if label_num == 2:
                img[img == 2] = 0       #turn edema to 0
                img[img != 0] = 1       #only keep necrosis, ET, NET = Tumor core
            if label_num == 4:
                img[img != 4] = 0       #only left ET
                img[img == 4] = 1
            if label_num == 3:
                img[img == 3] = 1       # remain GT, design for 2015 data
                
                
            img = img.astype('float32')
        else:
            img = (img-img.mean()) / img.std()      #normalization => zero mean   !!!care for the std=0 problem
            img = img.astype('float32')
        for slice in range(60,130):     #choose the slice range
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
            #img_g = augmentation(img_t,num_of_aug)
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n,:,:,:])
    
    return np.array(imgs)


#read one subject to show slices
count = 106
pul_seq = 'flair'
Flair = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*{}.nii.gz'.format(pul_seq), count, label=False)
pul_seq = 't1ce'
T1c = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*{}.nii.gz'.format(pul_seq), count, label=False)
pul_seq = 't1'
T1 = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*{}.nii.gz'.format(pul_seq), count, label=False)
pul_seq = 't2'
T2 = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*{}.nii.gz'.format(pul_seq), count, label=False)
label_num = 5
Label_full = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*seg.nii.gz', count, label=True)
label_num = 2
Label_core = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*seg.nii.gz', count, label=True)
label_num = 4
Label_ET = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*seg.nii.gz', count, label=True)
label_num = 3
Label_all = create_data_onesubject_val('./data/MICCAI_BraTS_2018_Data_Training/', '**/*seg.nii.gz', count, label=True)

predict_dataset = {"Flair": Flair, "T1c": T1c, "T1": T1, "T2": T2, "Label_full": Label_full, "Label_core": Label_core, "Label_ET": Label_ET, "Label_all": Label_all}


with open('predict_dataset.pickle', 'wb') as f:
    pickle.dump(predict_dataset, f)

print("predice_dataset has been generated.")
    
pul_seq = 'flair'
flair = create_data('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*{}.nii.gz'.format(pul_seq),label=False)
with open('Flair.pickle', 'wb') as f:
    pickle.dump(flair, f)


print("image type, flair,  has been generated.")
    
pul_seq = 't2'
T2 = create_data('./data/MICCAI_BraTS_2018_Data_Training/HGG/', '**/*{}.nii.gz'.format(pul_seq), label=False)
with open('T2.pickle', 'wb') as f:
    pickle.dump(T2, f)

print("image type, t2,  has been generated.")

label_num = 3
Label_all = create_data('./data/MICCAI_BraTS_2018_Data_Training/', '**/*seg.nii.gz', label=True)
np.save("Label_all.npy", Label_all)

print("label for whole tumor, has been generated.")