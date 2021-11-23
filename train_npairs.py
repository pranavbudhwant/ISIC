# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:18:11 2020

@author: prnvb
"""

from ae_datagenerator import get_encoder
from tensorflow.contrib.losses import metric_learning
from npair_generator import generator
from utils import EFNB4_INPUT_SHAPE
from keras.optimizers import Adam
from keras.layers import Input, Concatenate
from keras.models import Model

MODEL_NAME = 'efnb4'

def np_loss(batch_size):
    def loss(y_true,y_pred):
        labels = y_pred[:int(batch_size/2),-1]
        embeddings_anchor = y_pred[:int(batch_size/2),:-1]
        embeddings_positive = y_pred[int(batch_size/2):,:-1]
        return metric_learning.npairs_loss(labels=labels,
                           embeddings_anchor=embeddings_anchor,
                           embeddings_positive=embeddings_positive)
    return loss


batch_size = 8
gen = generator(batch_size=batch_size,
                model_name=MODEL_NAME,
                mode='train',image_size=EFNB4_INPUT_SHAPE[:-1],
                preprocess=True,
                onehotencode=False)

batch = next(gen)

encoder = get_encoder(MODEL_NAME)
encoder.trainable = True

label_input = Input(shape=(1,))
image_input = Input(shape=EFNB4_INPUT_SHAPE)
embedding = encoder(image_input)
out = Concatenate()([embedding,label_input])

model = Model([image_input,label_input],out)

model.summary()


#encoder.summary()

model.compile(Adam(0.001),np_loss(batch_size))

model.fit_generator(generator=gen,
                    steps_per_epoch=1,
                    epochs=1)


