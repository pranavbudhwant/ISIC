# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:22:46 2020

@author: prnvb
"""

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from keras.models import load_model
import joblib
from ae_datagenerator import get_encoder
from classifier_datagenerator import classifier_generator
from utils import LATENT_DIM, EFNB4_INPUT_SHAPE,ISIC_VAL_SAMPLES

# Random state we define this random state to use this value in TSNE which is a randmized algo.
RS = 25111993

def visualize_train_embeds(embeds, classes):
    #embeds, classes = generate_embeddings()
    embeds_2d = TSNE(random_state=RS).fit_transform(embeds)
    fig, ax = plt.subplots()
    scatter_x = embeds_2d[:,0]
    scatter_y = embeds_2d[:,1]
    group = classes
    for g in np.unique(group):
        i = np.where(group == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g)
    ax.legend()
    plt.show()


batch_size=32
encoder = get_encoder('efnb4')
generator = classifier_generator(batch_size=batch_size,
                                 model_name='efnb4',
                                 mode='val',
                                 image_size=EFNB4_INPUT_SHAPE[:-1],
                                 preprocess=True,
                                 onehotencode=False)

num_batches = int(ISIC_VAL_SAMPLES/batch_size)+1

emb = []; c = []
for nb in range(num_batches): #
    batch = next(generator)
    preds = encoder.predict_on_batch(batch[0])
    classes = batch[1]
    emb.append(preds)
    c.extend(list(classes))
    print(nb)
    
embeds = emb[0]
for e in emb[1:]:
    embeds = np.concatenate((embeds,e),axis=0)