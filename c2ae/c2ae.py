# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:08:16 2020

@author: prnvb
"""

from model import build_encoder, build_decoder_densenet, build_decoder_efnb2,\
                    build_decoder_efnb3, build_decoder_efnb4
from keras.layers import Dense, Input, Dropout, Multiply, Add, Concatenate
from keras.models import Model

from utils import LATENT_DIM,NUM_CLASSES,INPUT_SHAPE

def build_classifier(encoder,dropout_rate=0.4):
    input_image = Input(shape=INPUT_SHAPE)
    embedding = encoder(input_image)
    #out = Dense(int(LATENT_DIM/2),activation='relu')(embedding)
    if dropout_rate>0:
	    embedding = Dropout(0.3)(embedding)
    out = Dense(NUM_CLASSES,activation='softmax')(embedding)
    classifier = Model(input_image,out)
    classifier.name = 'Classifier'
    return classifier

def build_classifier_v2(encoder,input_shape):
    input_image = Input(shape=input_shape)
    embedding = encoder(input_image)
    #out = Dense(int(LATENT_DIM/2),activation='relu')(embedding)
    #out = Dropout(0.3)(out)
    out = Dense(NUM_CLASSES,activation='softmax')(embedding)
    classifier = Model(input_image,out)
    classifier.name = 'Classifier'
    return classifier


def build_conditioner():
    input_label_condition_vector = Input(shape=(NUM_CLASSES,))
    x = Dense(256,activation='relu')(input_label_condition_vector)
    #x = Dropout(0.2)(x)
    x = Dense(LATENT_DIM,activation='relu')(x)
    model = Model(input_label_condition_vector,x)
    return model

def build_c2ae(model_name): #encoder
    
    H_gamma = build_conditioner()
    H_gamma.name = 'H_gamma'
    H_beta = build_conditioner()
    H_beta.name = 'H_beta'
    
    #input_image = Input(shape=INPUT_SHAPE)
    #z = encoder(input_image)
    
    #condition_type_input = Input(shape=(1,))
    
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
    
    if model_name == 'densenet121':
        decoder = build_decoder_densenet(LATENT_DIM)
        
    if model_name == 'efnb2':
        decoder = build_decoder_efnb2(LATENT_DIM)
    
    if model_name == 'efnb3':
        decoder = build_decoder_efnb3(LATENT_DIM)
        
    if model_name == 'efnb4':
        decoder = build_decoder_efnb4(LATENT_DIM)
    
    match_recon = decoder(z_l_m)
    nonmatch_recon = decoder(z_l_nm)
    
    out = Concatenate(axis=-1)([match_recon,nonmatch_recon])
    
    #c2ae = Model(inputs=[input_image,l_j],outputs=reconstruction)
    #c2ae = Model(inputs=[z,l_j,condition_type_input],outputs=reconstruction)
    
    c2ae = Model(inputs=[z,l_m,l_nm],outputs=out)
    
    return c2ae, decoder, H_gamma, H_beta#, condition_type_input #, encoder

if __name__ == '__main__':
    encoder = build_encoder(LATENT_DIM)
    classifier = build_classifier(encoder)
    c2ae, _, decoder, H_gamma, H_beta = build_c2ae(encoder)
    
    encoder.summary()
    decoder.summary()
    H_gamma.summary()
    H_beta.summary()
    c2ae.summary()
