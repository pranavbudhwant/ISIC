# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:16:58 2020

@author: prnvb
"""

import os 
import cv2
import numpy as np
from keras.applications.densenet import preprocess_input

BASE_DIR = 'data/isic/ISIC_2019_Test_Input/'
image_paths = os.listdir(BASE_DIR)

def isictest_generator(batch_size,image_size):
    i=0
    while(1):
        if (i+1)*batch_size < len(image_paths):
            img_paths = image_paths[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            img_paths = image_paths[i*batch_size:]
            i=0
        imgs = [preprocess_input(np.array(cv2.resize(
                cv2.cvtColor(cv2.imread(BASE_DIR+x.rstrip()),cv2.COLOR_BGR2RGB),
                tuple(image_size)),dtype=np.float64)) for x in img_paths]
        imgs = np.array(imgs)
        yield(imgs)