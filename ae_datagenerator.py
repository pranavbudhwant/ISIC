# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:55:52 2020

@author: prnvb
"""

import cv2
import numpy as np
import pandas as pd
import joblib
import random
from sklearn.utils import shuffle
from keras.applications.densenet import preprocess_input
from keras.models import load_model
#import keras.backend as K

from model import build_encoder_v2
from c2ae import build_c2ae,build_classifier_v2
from utils import LATENT_DIM,EFNB4_INPUT_SHAPE,EFNB2_INPUT_SHAPE,\
                        DENSENET121_INPUT_SHAPE
from keras.applications.vgg16 import VGG16
from keras.models import Model

BASE_DIR = 'data/isic/'
IMAGE_BASE_DIR = BASE_DIR+'ISIC_2019_Training_Input/'
train_data = pd.read_csv(BASE_DIR+'train_gt.csv')
#data['X'] += '.jpg'
train_data = shuffle(train_data,random_state=10)
train_class_wise = joblib.load(BASE_DIR+'train_class_wise.pkl')

val_data = pd.read_csv(BASE_DIR+'val_gt.csv')
#data['X'] += '.jpg'
val_data = shuffle(val_data,random_state=10)
val_class_wise = joblib.load(BASE_DIR+'val_class_wise.pkl')

def get_encoder(model_name,ver='v1'):
    if model_name == 'efnb4':
        efnb4 = build_encoder_v2(LATENT_DIM,'efnb4')
        efnb4_classifier = build_classifier_v2(efnb4,EFNB4_INPUT_SHAPE)
        efnb4_classifier.load_weights('models/efnb4-v2-save-05-0.516.hdf5')
        encoder = efnb4_classifier.layers[1]
        encoder.trainable = False
        return encoder
    if model_name == 'efnb3':
        efnb3 = build_encoder_v2(LATENT_DIM,'efnb3')
        efnb3_classifier = build_classifier_v2(efnb3,EFNB2_INPUT_SHAPE)
        efnb3_classifier.load_weights('models/efnb3-v2-save-07-0.526.hdf5')
        encoder = efnb3_classifier.layers[1]
        encoder.trainable = False
        return encoder
    if model_name == 'efnb2':
        efnb2 = build_encoder_v2(LATENT_DIM,'efnb2')
        efnb2_classifier = build_classifier_v2(efnb2,EFNB2_INPUT_SHAPE)
        efnb2_classifier.load_weights('models/efnb2-v2-save-07-0.553.hdf5')
        encoder = efnb2_classifier.layers[1]
        encoder.trainable = False
        return encoder
    if model_name == 'densenet121':
        if ver=='v1':
            dnet121_classifier = load_model('models/densenet121-v2-save-04-0.59.hdf5')
            encoder = dnet121_classifier.layers[1]
            encoder.trainable = False
        else:
            dnet121_classifier = load_model('models/densenet121-v2-save-04-0.59.hdf5')
            enc = dnet121_classifier.layers[1]
            dnet = enc.layers[1]
            dnet.layers.pop()
            encoder = Model(dnet.inputs,dnet.layers[-1].output)
            encoder.trainable = False
        return encoder

encoder = get_encoder('densenet121','v2')
encoder._make_predict_function()

vgg = VGG16(include_top=False,weights='imagenet',
            input_shape=DENSENET121_INPUT_SHAPE)
vgg.trainable = False
for layer in vgg.layers:
    layer.trainable=False

vgg._make_predict_function()
selectedLayers = [2,9,17] #,5,13
selectedOutputs = [vgg.layers[i].output for i in selectedLayers]
lossModel = Model(vgg.inputs,selectedOutputs)
lossModel._make_predict_function()

def gen_label_condition_vecs(image_labels):
    batch_size = len(image_labels)
    l = np.ones((batch_size,8,)) * -1
    indxs = np.arange(0,batch_size)
    l[indxs,image_labels] = 1
    return l

def rangex(label):
    if label==0:
        return list(range(1,8))
    if label==7:
        return list(range(0,7))
    return list(range(0, label)) + list(range(label+1,8))

def gen_nm_labels(image_labels):
    nm_labels = []
    for m_label in image_labels:
        nm_labels.append(random.choice(rangex(m_label)))
    return nm_labels

def gen_nm_label_vecs(image_labels):
    nm_labels = gen_nm_labels(image_labels)
    return gen_label_condition_vecs(nm_labels), nm_labels

def autoencoder_generator(batch_size,image_size,encoder_name,mode='train',
                          subsample=1,
                          nonmatch_random=True,
                          recon_nm_isic_image = True,
                          test=False):
    i=0
    
    if mode == 'train':
        data = train_data.iloc[:int(subsample*len(train_data)),:]
        class_wise = train_class_wise
    else:
        data = val_data.iloc[:int(subsample*len(train_data)),:]
        class_wise = val_class_wise
    
    paths = list(data['X'])
    labels = list(data['Y_LE'])
    
    while 1:
        
        print(i)
        
        if (i+1)*batch_size < len(paths):
            image_paths = paths[i*batch_size:(i+1)*batch_size]
            image_labels = labels[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            image_paths = paths[i*batch_size:]
            image_labels = labels[i*batch_size:]
            i=0
        
        #Generate the matching condition vectors
        label_condition_vectors = gen_label_condition_vecs(image_labels)
        
        if nonmatch_random is True:
            #Generate the non-matching condition vectors
            nm_label_condition_vectors, _ = gen_nm_label_vecs(image_labels)
            #Generate random non-matching labels for sampling images:
            nm_labels = gen_nm_labels(image_labels)
        
        else:
            #Generate the non-matching condition vectors
            nm_label_condition_vectors, nm_labels = gen_nm_label_vecs(image_labels)
            
        if recon_nm_isic_image:
            #Select random image belonging to the nm_label
            nm_image_paths = []

            for nm_label in nm_labels:
                cw = class_wise[nm_label]
                nm_image_paths.append( random.choice(cw) )
            
            nm_images = [cv2.resize(cv2.cvtColor(
                    cv2.imread(IMAGE_BASE_DIR+x.rstrip()+'.jpg'),cv2.COLOR_BGR2RGB),
                                    image_size) for x in nm_image_paths]
        else:
            nm_images = []
            for q in range(len(image_paths)):
                nm_images.append( np.ones(DENSENET121_INPUT_SHAPE) )

        test_data = []

        images = [cv2.resize(cv2.cvtColor(
                cv2.imread(IMAGE_BASE_DIR+x.rstrip()),cv2.COLOR_BGR2RGB),
                                image_size) for x in image_paths]
        test_data.append(images)
        test_data.append(nm_images)
        test_data.append(image_paths)
        test_data.append(paths)

        images_encoder = []
        for image in images:
            images_encoder.append(preprocess_input(np.array(image,dtype=np.float64)))
        
        '''nm_images_encoder = nm_images
        nm_images_encoder = []
        for image in nm_images:
            nm_images_encoder.append(preprocess_input(np.array(image,dtype=np.float64)))
        '''

        images_decoder = []
        for image in images:
            images_decoder.append(image/255.)
        
        if recon_nm_isic_image:
            nm_images_decoder = []
            for image in nm_images:
                nm_images_decoder.append(image/255.)

        else:
            nm_images_decoder = nm_images

        mnm_concat_images_decoder = []
        for q in range(len(images)):
            mnm_concat_images_decoder.append(np.concatenate(
                    (images_decoder[q],nm_images_decoder[q]),axis=-1))
        
        z = encoder.predict_on_batch(np.array(images_encoder))
        
        if test:
            yield(test_data)
        else:
            yield([z, 
                label_condition_vectors, 
                nm_label_condition_vectors],
                np.array(mnm_concat_images_decoder))


def ae_perceptual_generator(batch_size,image_size,encoder_name,mode='train',
                          subsample=1,
                          nonmatch_random=True,
                          recon_nm_isic_image = True,
                          test=False):
    i=0
    
    if mode == 'train':
        data = train_data.iloc[:int(subsample*len(train_data)),:]
        class_wise = train_class_wise
    else:
        data = val_data.iloc[:int(subsample*len(train_data)),:]
        class_wise = val_class_wise
    
    paths = list(data['X'])
    labels = list(data['Y_LE'])
    
    while 1:
        
        #print(i)
        
        if (i+1)*batch_size < len(paths):
            image_paths = paths[i*batch_size:(i+1)*batch_size]
            image_labels = labels[i*batch_size:(i+1)*batch_size]
            i+=1
        else:
            image_paths = paths[i*batch_size:]
            image_labels = labels[i*batch_size:]
            i=0
        
        #Generate the matching condition vectors
        label_condition_vectors = gen_label_condition_vecs(image_labels)
        
        if nonmatch_random is True:
            #Generate the non-matching condition vectors
            nm_label_condition_vectors, _ = gen_nm_label_vecs(image_labels)
            #Generate random non-matching labels for sampling images:
            nm_labels = gen_nm_labels(image_labels)
        
        else:
            #Generate the non-matching condition vectors
            nm_label_condition_vectors, nm_labels = gen_nm_label_vecs(image_labels)
            
        if recon_nm_isic_image:
            #Select random image belonging to the nm_label
            nm_image_paths = []

            for nm_label in nm_labels:
                cw = class_wise[nm_label]
                nm_image_paths.append( random.choice(cw) )
            
            nm_images = [cv2.resize(cv2.cvtColor(
                    cv2.imread(IMAGE_BASE_DIR+x.rstrip()+'.jpg'),cv2.COLOR_BGR2RGB),
                                    image_size) for x in nm_image_paths]
        else:
            nm_images = []
            for q in range(len(image_paths)):
                nm_images.append( np.ones(DENSENET121_INPUT_SHAPE) )

        test_data = []

        images = [cv2.resize(cv2.cvtColor(
                cv2.imread(IMAGE_BASE_DIR+x.rstrip()),cv2.COLOR_BGR2RGB),
                                image_size) for x in image_paths]
        test_data.append(images)
        test_data.append(nm_images)
        test_data.append(image_paths)
        test_data.append(paths)

        images_encoder = []
        for image in images:
            images_encoder.append(preprocess_input(np.array(image,dtype=np.float64)))
        
        '''nm_images_encoder = nm_images
        nm_images_encoder = []
        for image in nm_images:
            nm_images_encoder.append(preprocess_input(np.array(image,dtype=np.float64)))
        '''

        images_decoder = []
        for image in images:
            images_decoder.append(image/255.)
        
        if recon_nm_isic_image:
            nm_images_decoder = []
            for image in nm_images:
                nm_images_decoder.append(image/255.)

        else:
            nm_images_decoder = nm_images
            
        '''
        mnm_concat_images_decoder = []
        for q in range(len(images)):
            mnm_concat_images_decoder.append(np.concatenate(
                    (images_decoder[q],nm_images_decoder[q]),axis=-1))
        '''
        match_activations = lossModel.predict_on_batch(np.array(images_decoder))
        nonmatch_activations = lossModel.predict_on_batch(np.array(nm_images_decoder))
        
        if len(selectedOutputs) > 1:
            mnm_concat_activations = [np.concatenate((match_activations[p],
                                      nonmatch_activations[p]),axis=-1) 
                                    for p in range(len(match_activations))]
        else:
            mnm_concat_activations = np.concatenate((match_activations,
                                      nonmatch_activations),axis=-1)
        
        z = encoder.predict_on_batch(np.array(images_encoder))
        
        if test:
            yield(test_data)
        else:
            yield([z, 
                label_condition_vectors, 
                nm_label_condition_vectors],
                mnm_concat_activations)


import matplotlib.pyplot as plt
def visualize_batch(batch):
    images = batch[0]
    nm_images = batch[1]
    fig = plt.figure(figsize=(12,12))
    columns = 2
    rows = len(batch[0])
    i=1
    for k in range(0,rows):
        img = images[k]
        fig.add_subplot(rows,columns,i)
        plt.imshow(img)
        img = nm_images[k]
        fig.add_subplot(rows,columns,i+1)
        plt.imshow(img)
        i+=2
    plt.show()

if __name__ == '__main__':
    
    batch_size = 4
    image_size = (224,224)
    encoder_name = 'densenet121'
    
    gen = autoencoder_generator(2,(224,224),
                                'densenet121',
                                'train')
    batch = next(gen)
    
    train_data_gen = autoencoder_generator(8,
                                     DENSENET121_INPUT_SHAPE[:-1],
                                     'densenet121',
                                     'train',
                                     1,
                                     nonmatch_random=False,
                                     test=True)
    
    train_data_gen = ae_perceptual_generator(8,
                                     DENSENET121_INPUT_SHAPE[:-1],
                                     'densenet121',
                                     'train',
                                     1,
                                     nonmatch_random=False,
                                     test=False)
    
    batch = next(train_data_gen)
    print(batch[-2])
    
    print(len(set(batch[-2])))
    
    
    