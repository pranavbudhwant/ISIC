# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:10:15 2020

@author: prnvb
"""

from keras.applications.resnet50 import ResNet50

from efficientnet.keras import EfficientNetB2, EfficientNetB3, EfficientNetB4
from keras.applications.densenet import DenseNet121

from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D,Input,\
                        Concatenate, Dense, Lambda, BatchNormalization,\
                        Reshape, UpSampling2D,Conv2D, Dropout, LeakyReLU,\
                        Conv2DTranspose

from keras.models import Model
import keras.backend as K
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

from utils import LATENT_DIM

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

def build_encoder(latent_dim,
                  model_name,
                  input_shape=(256,256,3), 
                  norm=False):
    model=None
    if model_name == 'resnet50':
        model = ResNet50(weights='imagenet',include_top=False,
                          input_shape=input_shape)
    elif model_name == 'efnb2':
        model = EfficientNetB2(weights='imagenet',include_top=False,
                          input_shape=input_shape)
    elif model_name == 'efnb4':
        model = EfficientNetB4(weights='imagenet',include_top=False,
                          input_shape=input_shape)
    elif model_name == 'densenet121':
        model = DenseNet121(weights='imagenet',include_top=False,
                          input_shape=input_shape)
    
    input_image = Input(shape=input_shape)
    model_out = model(input_image)
    model_out = Conv2D(int(latent_dim/2), kernel_size=(1,1), activation='relu')(model_out)
    avg_pool = GlobalAveragePooling2D()(model_out)
    max_pool = GlobalMaxPooling2D()(model_out)
    encoded = Concatenate()([avg_pool,max_pool])    
    
    '''
    avg_pool = GlobalAveragePooling2D()(model_out)
    max_pool = GlobalMaxPooling2D()(model_out)
    encoded = Concatenate()([avg_pool,max_pool])
    
    if norm:
        encoded = Dense(latent_dim)(encoded) #,activation='relu'
        encoded = Lambda(lambda  x: K.l2_normalize(x,axis=1))(encoded)
    else:
        encoded = Dense(latent_dim,activation='relu')(encoded)
    '''
    encoder = Model(inputs=input_image,outputs=encoded)
    encoder.name = 'Encoder'
    return encoder

def build_encoder_v2(latent_dim,model_name):
    model=None
    if model_name == 'resnet50':
        model = ResNet50(weights='imagenet',include_top=True)
        model.layers.pop()
        input_shape = (224,224,3)
        model2 = Model(model.input,model.layers[-1].output)
        model2.name = 'ResNet50'
    elif model_name == 'efnb2':
        model = EfficientNetB2(weights='imagenet',include_top=True)
        model.layers.pop()
        input_shape = (260,260,3)
        model2 = Model(model.input,model.layers[-1].output)
        model2.name = 'EfficientNetB2'
    elif model_name == 'efnb3':
        model = EfficientNetB3(weights='imagenet',include_top=True)
        model.layers.pop()
        input_shape = (300,300,3)
        model2 = Model(model.input,model.layers[-1].output)
        model2.name = 'EfficientNetB3'
    elif model_name == 'efnb4':
        model = EfficientNetB4(weights='imagenet',include_top=True)
        model.layers.pop()
        input_shape = (380,380,3)
        model2 = Model(model.input,model.layers[-1].output)
        model2.name = 'EfficientNetB4'
    elif model_name == 'densenet121':
        model = DenseNet121(weights='imagenet',include_top=True)
        model.layers.pop()
        input_shape = (224,224,3)
        model2 = Model(model.input,model.layers[-1].output)
        model2.name = 'DenseNet121'
    
    input_image = Input(shape=input_shape)
    model_out = model2(input_image)
    
    if model_name!='densenet121':
        model_out = Dense(latent_dim,activation='swish')(model_out)
    
    encoder = Model(inputs=input_image,outputs=model_out)
    encoder.name = 'Encoder'
    return encoder

#enc = build_encoder_v2(1024,'efnb3')
#efn = enc.layers[1]

#enc.summary()
#efn.summary()

def build_decoder_densenet(latent_dim):
    z_tensor = Input(shape=(latent_dim,))
    
    #x = Dense(2048,activation='relu')(z_tensor)
    #x = BatchNormalization()(x)
    
    x = Dense(3136,activation='relu')(z_tensor)
    x = BatchNormalization()(x)
    
    x = Reshape([7,7,64])(x)
    
    #up_samp = UpSampling2D(size = (2,2))(lr)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(64,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(16,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(3,(3,3), strides=(1,1),padding='same',activation='sigmoid')(x)
    
    #lr = Conv2DTranspose(64,7,strides=1,padding='valid')(lr)
    
    decoder = Model(z_tensor,x)
    decoder.name = 'DenseNetDecoder'

    return decoder

def build_decoder_efnb2(latent_dim):
    z_tensor = Input(shape=(latent_dim,))
    
    #x = Dense(2048,activation='relu')(z_tensor)
    #x = BatchNormalization()(x)
    
    x = Dense(2048,activation='relu')(z_tensor)
    x = BatchNormalization()(x)
    
    x = Reshape([4,4,128])(x)
    
    #up_samp = UpSampling2D(size = (2,2))(lr)
    #8
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    #16
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    #32
    x = Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    #64
    x = Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    #65
    x = Conv2DTranspose(64,(2,2),strides=(1,1),padding='valid')(x)
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    #130
    x = Conv2DTranspose(32,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    
    x = Conv2DTranspose(16,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(3,(3,3), strides=(1,1),padding='same',activation='sigmoid')(x)

    decoder = Model(z_tensor,x)
    decoder.name = 'EFNB2Decoder'

    return decoder



def build_decoder_efnb3(latent_dim):
    z_tensor = Input(shape=(latent_dim,))
    
    #x = Dense(2048,activation='relu')(z_tensor)
    #x = BatchNormalization()(x)
    
    x = Dense(2592,activation='relu')(z_tensor)
    x = BatchNormalization()(x)
    
    x = Reshape([9,9,32])(x)
    
    #up_samp = UpSampling2D(size = (2,2))(lr)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(64,(4,4),strides=(1,1),padding='valid')(x)
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(32,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(16,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(3,(3,3), strides=(1,1),padding='same',activation='sigmoid')(x)

    decoder = Model(z_tensor,x)
    decoder.name = 'EFNB3Decoder'

    return decoder

def build_decoder_efnb4(latent_dim):
    z_tensor = Input(shape=(latent_dim,))
    
    #x = Dense(2048,activation='relu')(z_tensor)
    #x = BatchNormalization()(x)
    
    x = Dense(1600,activation='relu')(z_tensor)
    x = BatchNormalization()(x)
    
    x = Reshape([5,5,64])(x)
    
    #up_samp = UpSampling2D(size = (2,2))(lr)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(256,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(128,(6,6),strides=(1,1),padding='valid')(x)
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(128,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    #90
    x = Conv2DTranspose(64,(6,6),strides=(1,1),padding='valid')(x)
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(32,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32,(3,3),strides=(1,1),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2DTranspose(16,(3,3),strides=(2,2),padding='same')(x)
    x = Conv2D(3,(3,3), strides=(1,1),padding='same',activation='sigmoid')(x)

    decoder = Model(z_tensor,x)
    decoder.name = 'EFNB4Decoder'

    return decoder


if __name__ == '__main__':
    #encoder = build_encoder(LATENT_DIM)
    #encoder.summary()
    
    decoder = build_decoder_efnb2(LATENT_DIM)
    decoder.summary()
    