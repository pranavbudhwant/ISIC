# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:26:08 2020

@author: prnvb
"""

from keras.models import load_model
from model import build_encoder_v2
from c2ae import build_classifier_v2
from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,\
                            f1_score,balanced_accuracy_score,roc_auc_score
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input as densenet_preprocess_input


MODEL = 'efnb4'
INPUT_SHAPE = EFNB4_INPUT_SHAPE
encoder = build_encoder_v2(LATENT_DIM,MODEL)
classifier = build_classifier_v2(encoder,INPUT_SHAPE)
classifier.load_weights('models/efnb4-v2-save-05-0.516.hdf5')

BASE_DIR = 'data/isic/'
df = pd.read_csv(BASE_DIR+'ISIC_2019_Training_GroundTruth.csv')
df['X']=df['image']+'.jpg'
df['Y'] = df.iloc[:,1:10].idxmax(axis=1)
df.head()

train_df,val_df = train_test_split(df,test_size=0.2,random_state=10)
val_df.head()

y_true = val_df['Y']

batch_size = 32

datagen = ImageDataGenerator(vertical_flip=True, horizontal_flip=True,
                             height_shift_range = 0.20, width_shift_range=0.20, 
                             rotation_range=20,
                             preprocessing_function=densenet_preprocess_input)

val_data_generator = datagen.flow_from_dataframe(val_df,
                                    directory=BASE_DIR+'ISIC_2019_Training_Input',
                                    x_col='X',y_col='Y',
                                    target_size=INPUT_SHAPE[:-1],
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    shuffle=True,seed=1,
                                    interpolation='lanczos')

y_pred = []
batch = next(val_data_generator)
preds = classifier.predict_on_batch(batch)
y_pred.append(  )