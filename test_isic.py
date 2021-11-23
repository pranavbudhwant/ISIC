# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:39:33 2020

@author: prnvb
"""

from c2ae import build_classifier_v2, build_c2ae
from utils import LATENT_DIM,EFNB4_INPUT_SHAPE,EFNB2_INPUT_SHAPE,\
                        EFNB3_INPUT_SHAPE,DENSENET121_INPUT_SHAPE,NUM_CLASSES
from model import build_encoder_v2
from keras.models import load_model, Model
from keras.layers import Input,Add,Multiply
from openset_test_generator import openset_generator
import numpy as np

from sklearn.metrics import accuracy_score,precision_score,recall_score,\
                            f1_score,balanced_accuracy_score,roc_auc_score,\
                            confusion_matrix

import joblib
import matplotlib.pyplot as plt

import keras.backend as K
K.tensorflow_backend._get_available_gpus()

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

def gen_label_condition_vecs(image_labels):
    batch_size = len(image_labels)
    l = np.ones((batch_size,8,)) * -1
    indxs = np.arange(0,batch_size)
    l[indxs,image_labels] = 1
    return l

def l2_numpy_loss(y_true,y_pred):
    return np.mean(np.square(y_true-y_pred))

densenet_classifier = load_model('models/densenet121-v2-save-04-0.59.hdf5')
efnb2_classifier = build_classifier_v2(build_encoder_v2(LATENT_DIM,'efnb2'),
                                       EFNB2_INPUT_SHAPE)
efnb3_classifier = build_classifier_v2(build_encoder_v2(LATENT_DIM,'efnb3'),
                                       EFNB3_INPUT_SHAPE)
efnb4_classifier = build_classifier_v2(build_encoder_v2(LATENT_DIM,'efnb4'),
                                       EFNB4_INPUT_SHAPE)

#densenet_classifier.load_weights('models/densenet121-v2-save-04-0.59.hdf5')
efnb2_classifier.load_weights('models/efnb2-v2-save-07-0.553.hdf5')
efnb3_classifier.load_weights('models/efnb3-v2-save-07-0.526.hdf5')
efnb4_classifier.load_weights('models/efnb4-v2-save-05-0.516.hdf5')

c2ae_perceptual = build_c2ae('densenet121','v2')[-1]
c2ae_perceptual.load_weights('models/c2ae/v2c2ae-densenet-white-save-0.8-04-7.576-8.400.hdf5')
H_gamma = c2ae_perceptual.layers[3]
H_gamma.trainable=False
H_beta = c2ae_perceptual.layers[5]
H_beta.trainable=False
DenseNetDecoder = c2ae_perceptual.layers[9]
DenseNetDecoder.trainable=False
DenseNetEncoder = get_encoder('densenet121','v2')
input_image = Input(shape=DENSENET121_INPUT_SHAPE)
z = DenseNetEncoder(input_image)
l = Input(shape=(NUM_CLASSES,))
gamma = H_gamma(l)
beta = H_beta(l)
z_l = Multiply()([z,gamma])
z_l = Add()([z_l,beta])
reconstruction = DenseNetDecoder(z_l)
c2ae = Model(inputs=[input_image,l],outputs=reconstruction)

c2ae.summary()

match_recon = joblib.load('recon_errors/v2densenet-0.8-04-match_recon.pkl')
nonmatch_recon = joblib.load('recon_errors/v2densenet-0.8-04-nonmatch_recon.pkl')

plt.hist(match_recon)
plt.hist(nonmatch_recon)


RECON_THRESHOLD = 0.03

batch_size=32
nb = int(np.ceil(11275/batch_size))
os_generator = openset_generator(batch_size,DENSENET121_INPUT_SHAPE[:-1],0)

y_true_labels = []
y_pred_labels = []

for b in range(nb):
    batch = next(os_generator)
    preds=densenet_classifier.predict_on_batch(batch[0])
    y_pred = []
    y_pred_prob = []
    for i in range(len(preds)):
        y_pred.append(np.argmax(preds[i,:]))
        y_pred_prob.append(preds[i,5])
    y_pred_cond_vecs = gen_label_condition_vecs(y_pred)
    recons = c2ae.predict_on_batch([batch[0],y_pred_cond_vecs])
    decoder_images = batch[1][0]
    recon_errors = []
    for i in range(len(y_pred)):
        re = l2_numpy_loss(decoder_images[i,:,:,:],recons[i,:,:,:])
        recon_errors.append(re)
        if re>RECON_THRESHOLD or y_pred_prob[i]<0.7:
            y_pred[i] = 8
    y_true_labels.extend(batch[1][2])
    y_pred_labels.extend(y_pred)
    print(b)


print('Normalized Accuracy: ',accuracy_score(y_true_labels,y_pred_labels))
print('Precision Score    : ',precision_score(y_true_labels,y_pred_labels, average='macro'))
print('Recall Score       : ',recall_score(y_true_labels,y_pred_labels, average='macro'))
print('F1 Score           : ',f1_score(y_true_labels,y_pred_labels,average='macro'))
print('Balanced Accuracy  : ',balanced_accuracy_score(y_true_labels,y_pred_labels))


confusion_matrix(y_true_labels,y_pred_labels)
