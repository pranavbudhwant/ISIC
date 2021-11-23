# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:40:35 2020

@author: prnvb
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import ceil
import joblib
import random

from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input

from utils import NUM_CLASSES, LATENT_DIM

DS = 'ISIC' #SD128

BASE_DIR = 'data/isic/'
IMAGE_BASE_DIR = BASE_DIR+'ISIC_2019_Training_Input/'
train_data = pd.read_csv(BASE_DIR+'train_gt.csv')
val_data = pd.read_csv(BASE_DIR+'val_gt.csv')

train_classwise = joblib.load('data/isic/train_class_wise.pkl')
val_classwise = joblib.load('data/isic/val_class_wise.pkl')

train_images = list(train_data['X'])
train_image_labels = list(train_data['Y_LE'])

val_images = list(val_data['X'])
val_image_labels = list(val_data['Y_LE'])

preprocessors = {}
preprocessors['resnet50'] = resnet_preprocess_input
preprocessors['densenet121'] = densenet_preprocess_input
preprocessors['efnb2'] = densenet_preprocess_input
preprocessors['efnb4'] = densenet_preprocess_input

def generator(batch_size,model_name,
             mode='train',
             image_size=tuple((256,256)),
             preprocess=True,
             onehotencode=True,
             subsample=1):
    
    preprocess_input = preprocessors[model_name]
    if mode == 'train':
        images = train_images[:int(subsample*len(train_images))]
        image_labels = train_image_labels[:int(subsample*len(train_images))]
        classwise = train_classwise
    else:
        images = val_images[:int(subsample*len(val_images))]
        image_labels = val_image_labels[:int(subsample*len(val_images))]
        classwise = val_classwise

    i=0
    while(1):
        if (i+1)*int(batch_size/2) < len(images):
            img_paths = images[i*int(batch_size/2):(i+1)*int(batch_size/2)]
            img_labels = image_labels[i*int(batch_size/2):(i+1)*int(batch_size/2)]
            i+=1
        else:
            img_paths = images[i*int(batch_size/2):]
            img_labels = image_labels[i*int(batch_size/2):]
            i=0
        
        positive_img_paths = []
        for l in img_labels:
            positive_img_paths.append( random.choice(classwise[l]) + '.jpg')
        
        img_labels.extend(img_labels)
        img_paths.extend(positive_img_paths)
        
        if preprocess:
            imgs = [preprocess_input(np.array(cv2.resize(
                    cv2.cvtColor(cv2.imread(IMAGE_BASE_DIR+x.rstrip()),cv2.COLOR_BGR2RGB),
                    tuple(image_size)),dtype=np.float64)) for x in img_paths]
            imgs = np.array(imgs)
        else:
            imgs = [cv2.resize(cv2.cvtColor(cv2.imread(
                    IMAGE_BASE_DIR+x.rstrip()),cv2.COLOR_BGR2RGB),image_size) 
                    for x in img_paths]
            imgs = np.array(imgs)
            imgs = imgs/255. #(imgs - 127.5)/127.5
        
        if onehotencode:
            y = np.zeros((len(imgs),NUM_CLASSES))
            for i in range(len(img_labels)):
                y[i,img_labels[i]] = 1
        else:
            y = np.array(img_labels)
        
        dummy_y = np.zeros((len(y),LATENT_DIM+1))
        
        yield([imgs,y],dummy_y)

if __name__ == '__main__':
    gen = generator(batch_size=16,
                    model_name='efnb4',
                    mode='train',
                    image_size=(380,380),
                    preprocess=True,
                    onehotencode=False,
                    subsample=0.5)

    for i in range(1266):
        batch = next(gen)
    