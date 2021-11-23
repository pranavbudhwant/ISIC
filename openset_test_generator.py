# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:16:17 2020

@author: prnvb
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import ceil
from keras.applications.densenet import preprocess_input
from utils import NUM_CLASSES

openset_data = pd.read_csv('data/openset_data.csv')
openset_data['Y_LE_ASC'] = 8
ISIC_BASE_DIR = 'data/isic/ISIC_2019_Training_Input/'
SD_BASE_DIR = 'data/sd198/sd-198/'
mapp = {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'VASC': 7}
for i in range(len(openset_data)):
    if openset_data['DS'][i] == 'ISIC':
        openset_data['X'][i] = ISIC_BASE_DIR + openset_data['X'][i]
        openset_data['Y_LE_ASC'][i] = mapp[openset_data['Y'][i]]
    else:
        openset_data['X'][i] = SD_BASE_DIR + openset_data['X'][i]

def openset_generator(batch_size,
                      image_size=(256,256),
                      starti=0,
                      test=False):
    i=starti
    image_paths = list(openset_data['X'])
    image_classes = list(openset_data['Y'])
    image_labels = list(openset_data['Y_LE_ASC'])
    
    while(1):
        test_data=[]
        if (i+1)*batch_size < len(image_paths):
            img_paths = image_paths[i*batch_size:(i+1)*batch_size]
            img_classes = image_classes[i*batch_size:(i+1)*batch_size]
            img_labels = image_labels[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            img_paths = image_paths[i*batch_size:]
            img_classes = image_classes[i*batch_size:]
            img_labels = image_labels[i*batch_size:]
            i=0
        
        test_data.append(img_paths)
        test_data.append(img_classes)
        
        if not test:
            encoder_imgs = [preprocess_input(np.array(cv2.resize(
                    cv2.cvtColor(cv2.imread(x.rstrip()),cv2.COLOR_BGR2RGB),
                    tuple(image_size)),dtype=np.float64)) for x in img_paths]
            encoder_imgs = np.array(encoder_imgs)
    
            decoder_imgs = [cv2.resize(cv2.cvtColor(cv2.imread(
                    x.rstrip()),cv2.COLOR_BGR2RGB),image_size) 
                    for x in img_paths]
            decoder_imgs = np.array(decoder_imgs)
            decoder_imgs = decoder_imgs/255. #(imgs - 127.5)/127.5
        
            yield(encoder_imgs, [decoder_imgs,img_classes,img_labels])
        else:
            yield(test_data)

if __name__ == '__main__':
    os_generator = openset_generator(32,starti=332,test=False)
    batch = next(os_generator)
    nb = 352
    for i in range(nb):
        batch = next(os_generator)
    
