# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:35:45 2020

@author: prnvb
"""

import pandas as pd
from model import build_encoder
from c2ae import build_classifier
from classifier_datagenerator import classifier_generator
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras_preprocessing.image import ImageDataGenerator

import keras.backend as K
K.tensorflow_backend._get_available_gpus()


from sklearn.model_selection import train_test_split

from utils import INPUT_SHAPE, LATENT_DIM, ISIC_TRAIN_SAMPLES,ISIC_VAL_SAMPLES


MODEL = 'efnb2'

encoder = build_encoder(LATENT_DIM,MODEL)
classifier = build_classifier(encoder)

classifier.summary()

classifier.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',
                   metrics=['categorical_accuracy'])

BASE_DIR = 'data/isic/'
df = pd.read_csv(BASE_DIR+'ISIC_2019_Training_GroundTruth.csv')
df['X']=df['image']+'.jpg'
df['Y'] = df.iloc[:,1:10].idxmax(axis=1)
df.head()

train_df,val_df = train_test_split(df,test_size=0.2,random_state=10)
val_df.head()

datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True,
                             height_shift_range = 0.20, width_shift_range=0.20, 
                             rotation_range=20,
                             preprocessing_function=densenet_preprocess_input)


batch_size = 1
epochs = 1

train_g = datagen.flow_from_dataframe(train_df,
                                      directory=BASE_DIR+'ISIC_2019_Training_Input',
                                      x_col='X',y_col='Y',
                                      target_size=INPUT_SHAPE[:-1],
                                      class_mode='categorical',
                                      batch_size=batch_size,
                                      shuffle=True,seed=1,
                                      interpolation='lanczos')

val_g = datagen.flow_from_dataframe(val_df,
                                    directory=BASE_DIR+'ISIC_2019_Training_Input',
                                    x_col='X',y_col='Y',
                                    target_size=INPUT_SHAPE[:-1],
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle=True,seed=1,
                                    interpolation='lanczos')


train_data_generator = classifier_generator(batch_size,MODEL,mode='train')
val_data_generator = classifier_generator(batch_size,MODEL,mode='val')

train_nb = int(ISIC_TRAIN_SAMPLES/batch_size)+1
val_nb = int(ISIC_VAL_SAMPLES/batch_size)+1

filepath='models/'+MODEL+"-save-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
                             save_weights_only=False,
                             verbose=1, save_best_only=True, mode='min',
                             period=1)
callbacks_list = [checkpoint]

train_history = classifier.fit_generator(train_g,
                                         steps_per_epoch=train_nb,
                                         epochs=epochs,
                                         validation_data = val_g,
                                         validation_steps = val_nb,
                                         callbacks=callbacks_list)
