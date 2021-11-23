# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:08:16 2020

@author: prnvb
"""

from model import build_encoder, build_decoder_densenet, build_decoder_efnb2,\
                    build_decoder_efnb3, build_decoder_efnb4,build_decoder_densenet_v2
from keras.layers import Dense, Input, Dropout, Multiply, Add, Concatenate,Reshape
from keras.models import Model
from keras.applications.vgg16 import VGG16
from utils import LATENT_DIM,NUM_CLASSES,DENSENET121_INPUT_SHAPE,\
EFNB2_INPUT_SHAPE,EFNB3_INPUT_SHAPE,EFNB4_INPUT_SHAPE

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

def build_conditioner_v2():
    input_label_condition_vector = Input(shape=(NUM_CLASSES,))
    x = Dense(7*7*1024,activation='relu')(input_label_condition_vector)
    x = Reshape([7,7,1024])(x)
    model = Model(input_label_condition_vector,x)
    return model

def build_c2ae(model_name,version='v1'): #encoder
    
    if version=='v1':
        H_gamma = build_conditioner()
        H_gamma.name = 'H_gamma'
        H_beta = build_conditioner()
        H_beta.name = 'H_beta'
        z = Input(shape=(LATENT_DIM,))
    else:
        H_gamma = build_conditioner_v2()
        H_gamma.name = 'H_gamma'
        H_beta = build_conditioner_v2()
        H_beta.name = 'H_beta'
        z = Input(shape=(7,7,1024))
    #input_image = Input(shape=INPUT_SHAPE)
    #z = encoder(input_image)
    
    #condition_type_input = Input(shape=(1,))
    
    
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
    
    if version=='v1':
        decoder = build_decoder_densenet(LATENT_DIM)
    else:
        decoder = build_decoder_densenet_v2()
        
    vgg = VGG16(include_top=False,weights='imagenet',
                input_shape=DENSENET121_INPUT_SHAPE)
        
    if model_name == 'efnb2':
        decoder = build_decoder_efnb2(LATENT_DIM)
        vgg = VGG16(include_top=False,weights='imagenet',
                    input_shape=EFNB2_INPUT_SHAPE)
    
    if model_name == 'efnb3':
        decoder = build_decoder_efnb3(LATENT_DIM)
        vgg = VGG16(include_top=False,weights='imagenet',
                    input_shape=EFNB3_INPUT_SHAPE)
        
    if model_name == 'efnb4':
        decoder = build_decoder_efnb4(LATENT_DIM)
        vgg = VGG16(include_top=False,weights='imagenet',
                    input_shape=EFNB4_INPUT_SHAPE)
    
    vgg.trainable = False
    for layer in vgg.layers:
        layer.trainable=False
    
    match_recon = decoder(z_l_m)
    nonmatch_recon = decoder(z_l_nm)
    
    out = Concatenate(axis=-1)([match_recon,nonmatch_recon])
    
    c2ae = Model(inputs=[z,l_m,l_nm],outputs=[match_recon,nonmatch_recon])
    
    selectedLayers = [2,9,17] #,5,13
    selectedOutputs = [vgg.layers[i].output for i in selectedLayers]
    lossModel = Model(vgg.inputs,selectedOutputs)
    
    MR_lossModelOutputs = lossModel(match_recon)
    NMR_lossModelOutputs = lossModel(nonmatch_recon)
    if len(selectedOutputs) > 1:
        perceptual_out = [Concatenate(axis=-1)([MR_lossModelOutputs[i],
                                               NMR_lossModelOutputs[i]]) 
                                for i in range(len(MR_lossModelOutputs))]
    else:
        perceptual_out = Concatenate(axis=-1)([MR_lossModelOutputs,
                                               NMR_lossModelOutputs])
    c2ae_perceptual = Model(c2ae.inputs,perceptual_out) #[z,l_m,l_nm]
    
    #c2ae = Model(inputs=[input_image,l_j],outputs=reconstruction)
    #c2ae = Model(inputs=[z,l_j,condition_type_input],outputs=reconstruction)
    
    c2ae_con = Model(inputs=[z,l_m,l_nm],outputs=out)
    
    return c2ae_con, decoder, H_gamma, H_beta,c2ae_perceptual#, condition_type_input #, encoder

if __name__ == '__main__':
    #encoder = build_encoder(LATENT_DIM,'densenet121')
    #encoder = get_encoder('densenet121','v2')
    #classifier = build_classifier(encoder)
    c2ae, decoder, H_gamma, H_beta, c2ae_perceptual = build_c2ae('densenet121','v2')
    
    #encoder.summary()
    decoder.summary()
    H_gamma.summary()
    H_beta.summary()
    c2ae.summary()
