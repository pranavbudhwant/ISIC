# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:47:10 2020

@author: prnvb
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import ceil

from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input

from utils import NUM_CLASSES

DS = 'ISIC' #SD128

if DS == 'SD128':
    BASE_DIR = 'data/sd198/sd-198/'
    IMAGE_BASE_DIR = BASE_DIR + 'images/'
    with open(BASE_DIR+'image_class_labels.txt', 'r') as f:
        image_labels = f.readlines()
        image_labels = [int(x.split(' ')[1]) for x in image_labels]
    with open(BASE_DIR+'images.txt', 'r') as f:
        images = f.readlines()
        images = [x.split(' ')[1] for x in images]
    club = list(zip(images,image_labels))
    club = shuffle(club,random_state=10)
    images,image_labels = zip(*club)

elif DS == 'ISIC':
    BASE_DIR = 'data/isic/'
    IMAGE_BASE_DIR = BASE_DIR+'ISIC_2019_Training_Input/'
    train_data = pd.read_csv(BASE_DIR+'train_gt.csv')
    val_data = pd.read_csv(BASE_DIR+'val_gt.csv')
    #data = shuffle(data,random_state=10)
    #train_data['image'] += '.jpg'
    #val_data['image'] += '.jpg'

    train_images = list(train_data['X'])
    train_image_labels = list(train_data['Y_LE'])

    val_images = list(val_data['X'])
    val_image_labels = list(val_data['Y_LE'])

preprocessors = {}
preprocessors['resnet50'] = resnet_preprocess_input
preprocessors['densenet121'] = densenet_preprocess_input
preprocessors['efnb2'] = densenet_preprocess_input
preprocessors['efnb4'] = densenet_preprocess_input

def classifier_generator(batch_size,model_name,
                         mode='train',
                         image_size=tuple((256,256)),
                         preprocess=True,
                         onehotencode=True):
    
    preprocess_input = preprocessors[model_name]
    if mode == 'train':
        images = train_images
        image_labels = train_image_labels
    else:
        images = val_images
        image_labels = val_image_labels

    i=0
    while(1):
        if (i+1)*batch_size < len(images):
            img_paths = images[i*batch_size:(i+1)*batch_size]
            img_labels = image_labels[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            img_paths = images[i*batch_size:]
            img_labels = image_labels[i*batch_size:]
            i=0
        
        
        if preprocess:
            #print(image_size)
            #print(type(image_size))
            #
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
            if DS == 'SD128':
                sub = 1
            else:
                sub = 0
            y = np.zeros((len(imgs),NUM_CLASSES))
            for i in range(len(img_labels)):
                y[i,img_labels[i]-sub] = 1
        else:
            y = np.array(img_labels)
        
        yield(imgs,y)

def image_generator(batch_size,model_name,
                         mode='train',
                         image_size=tuple((256,256)),
                         preprocess=True,
                         onehotencode=True):
    
    preprocess_input = preprocessors[model_name]
    if mode == 'train':
        images = train_images
    else:
        images = val_images
    i=0
    while(1):
        if (i+1)*batch_size < len(images):
            img_paths = images[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            img_paths = images[i*batch_size:]
            i=0
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
        
        yield(imgs)
        

def class_generator(batch_size,model_name,
                     mode='train',
                     image_size=tuple((256,256)),
                     preprocess=True,
                     onehotencode=True):
    if mode == 'train':
        image_labels = train_image_labels
    else:
        image_labels = val_image_labels

    i=0
    while(1):
        if (i+1)*batch_size < len(images):
            img_labels = image_labels[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            img_labels = image_labels[i*batch_size:]
            i=0
        
        if onehotencode:
            if DS == 'SD128':
                sub = 1
            else:
                sub = 0
            y = np.zeros((len(img_labels),NUM_CLASSES))
            for i in range(len(img_labels)):
                y[i,img_labels[i]-sub] = 1
        else:
            y = np.array(img_labels)
        yield(y)


def visualize_batch(batch):
    images = batch[0]
    fig = plt.figure(figsize=(8,8))
    columns = 4
    rows = int(ceil(len(batch[0])/columns))
    
    for i in range(1,columns*rows+1):
        img = images[i-1,:,:,:]
        fig.add_subplot(rows,columns,i)
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':

    train_data_generator = classifier_generator(batch_size=32,
                                                model_name='efnb2',
                                                mode='train',
                                                preprocess=False)
    batch = next(train_data_generator)
    visualize_batch(batch)

    val_data_generator = classifier_generator(batch_size=32,
                                                model_name='efnb2',
                                                mode='val')
    batch = next(val_data_generator)
    
    
    
    #for i in range( int(6584/32)+5 ):
    #    batch = next(data_generator)