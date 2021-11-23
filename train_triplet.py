# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:42:58 2020

@author: prnvb
"""

from model import build_encoder
from datagenerator import generator
from tensorflow.contrib.losses import metric_learning
from keras.layers import Input,Concatenate
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
from tf_triplet_loss import batch_all_triplet_loss,\
                            batch_hard_triplet_loss

def triplet_loss(y_true,y_pred):
    labels = y_pred[:,0]
    embeds = y_pred[:,1:]

    return batch_hard_triplet_loss(labels=labels,
                                  embeddings=embeds,
                                  margin=10.)

    '''
    return batch_all_triplet_loss(labels=labels,
                                  embeddings=embeds,
                                  margin=10.)
    return metric_learning.triplet_semihard_loss(labels=labels,
                                                 embeddings=embeds,
                                                 margin=2.)
    '''
    
    
def build_model(latent_dim):
    input_image = Input(shape=(224,224,3))
    encoder = build_encoder(latent_dim=latent_dim)
    #encoder.summary()
    embedding = encoder(input_image)
    input_class = Input(shape=(1,))
    concatenated_output = Concatenate()([input_class,embedding])
    model = Model(inputs=[input_class,input_image],outputs=concatenated_output)
    return model

batch_size = 1
latent_dim = 512
total_samples = 6584
dummy_y = np.zeros((batch_size,latent_dim+1))

model = build_model(latent_dim)
data_generator = generator(batch_size)

#model.summary()

model.compile(optimizer=Adam(0.0001),loss=triplet_loss)

num_epochs = 1
num_batches = int(total_samples/batch_size)+1

losses = []
for n_epoch in range(num_epochs):
    for n_batch in range(num_batches):
        batch = next(data_generator)
        loss = model.train_on_batch([batch[0],batch[1]],dummy_y)
        losses.append(loss)
        print(loss)
