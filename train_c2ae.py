# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:14:45 2020

@author: prnvb
"""

import numpy as np
from c2ae import build_c2ae
import tensorflow as tf
from ae_datagenerator import *
from utils import LATENT_DIM,EFNB4_INPUT_SHAPE,EFNB2_INPUT_SHAPE,\
                        DENSENET121_INPUT_SHAPE, NUM_CLASSES, LATENT_DIM
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Input, Add, Multiply, Concatenate
from keras.models import Model

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true-y_pred))

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred))
    #return tf.reduce_mean(tf.squared_difference(y_true, y_pred))

def l2_loss_np(y_true, y_pred):
    return np.mean(np.square(y_true-y_pred))

def c2ae_loss(alpha=0.9): #condition_type
    #condition_type -> Matching/Non Matching
    #If matching, then 1, else 0.
    def custom_l1_loss(y_true,y_pred):
        #return condition_type
        
        match_image = y_true[:,:,:3]
        nonmatch_image = y_true[:,:,3:]
        
        match_recon = y_pred[:,:,:3]
        nonmatch_recon = y_pred[:,:,3:]
        
        return alpha*l1_loss(match_image,match_recon) + \
                (1-alpha)*l1_loss(nonmatch_image,nonmatch_recon)
        
    return custom_l1_loss

def c2ae_perceptual_loss(alpha=0.9):
    def custom_l1_loss(y_true,y_pred):
        nc = int(K.int_shape(y_pred)[-1]/2)
        match_loss = l2_loss(y_true[:,:,:,:nc],y_pred[:,:,:,:nc])
        nonmatch_loss = l2_loss(y_true[:,:,:,nc:],y_pred[:,:,:,nc:])
        return alpha*match_loss + (1-alpha)*nonmatch_loss
    return custom_l1_loss

ENCODER_NAME = 'densenet121'

c2ae = build_c2ae(ENCODER_NAME)[-1] #,_,_,_,cinp 
c2ae.summary()

c2ae.compile(optimizer=Adam(1e-5),loss=c2ae_perceptual_loss(0.9))


batch_size = 2
num_batches = 10#int(25333/batch_size)+1
'''
train_data_gen = autoencoder_generator(batch_size,
                                     DENSENET121_INPUT_SHAPE[:-1],
                                     ENCODER_NAME,
                                     'train')

val_data_gen = autoencoder_generator(batch_size,
                                     DENSENET121_INPUT_SHAPE[:-1],
                                     ENCODER_NAME,
                                     'val')
'''

train_data_gen = ae_perceptual_generator(batch_size,
                                 DENSENET121_INPUT_SHAPE[:-1],
                                 ENCODER_NAME,
                                 'train',
                                 1,
                                 nonmatch_random=False,
                                 recon_nm_isic_image = False,
                                 test=False)

val_data_gen = ae_perceptual_generator(batch_size,
                                 DENSENET121_INPUT_SHAPE[:-1],
                                 ENCODER_NAME,
                                 'train',
                                 1,
                                 nonmatch_random=False,
                                 recon_nm_isic_image = False,
                                 test=False)


batch = next(train_data_gen)

o = c2ae.predict_on_batch(batch[0])


c2ae.fit_generator(generator=train_data_gen,
                   steps_per_epoch=num_batches,
                   epochs=20,
                   validation_data=val_data_gen,
                   validation_steps=num_batches)


H_gamma = c2ae.layers[3]
H_gamma.trainable=False
H_beta = c2ae.layers[5]
H_beta.trainable=False
DenseNetDecoder = c2ae.layers[9]
DenseNetDecoder.trainable=False

z = Input(shape=(LATENT_DIM,))

l_m = Input(shape=(NUM_CLASSES,))
gamma_m = H_gamma(l_m)
beta_m = H_beta(l_m)
z_l_m = Multiply()([z,gamma_m])
z_l_m = Add()([z_l_m,beta_m])


l_nm = Input(shape=(NUM_CLASSES,))
gamma_nm = H_gamma(l_nm)
beta_nm = H_beta(l_nm)
z_l_nm = Multiply()([z,gamma_nm])
z_l_nm = Add()([z_l_nm,beta_nm])

match_recon = DenseNetDecoder(z_l_m)
nonmatch_recon = DenseNetDecoder(z_l_nm)

out = Concatenate(axis=-1)([match_recon,nonmatch_recon])

c2ae_test = Model(inputs=[z,l_m,l_nm],outputs=out)#[match_recon,nonmatch_recon]
c2ae_test.summary()



losses = []
def train(num_epochs):
    data_gen = autoencoder_generator(batch_size,
                                     DENSENET121_INPUT_SHAPE[:-1],
                                     ENCODER_NAME,
                                     'train')
    for ne in range(num_epochs):
        for nb in range(num_batches):
            
            mnm_concat_images_decoder,z, \
            label_condition_vectors, \
            nm_label_condition_vectors = next(data_gen)
            
            #z = encoder.predict_on_batch(images_encoder)
            
            loss = c2ae.train_on_batch([z,
                                        label_condition_vectors,
                                        nm_label_condition_vectors],
                                        mnm_concat_images_decoder)
            losses.append(loss)

train(1)
