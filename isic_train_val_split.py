# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:05:25 2020

@author: prnvb
"""

import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv('data/isic/labelEncoded.csv')

train_size = 0.8
idx = int(len(data)*train_size)+1

data = shuffle(data,random_state=10)

train_data = data.iloc[:idx,:]
val_data = data.iloc[idx:,:]

train_data.to_csv('data/isic/labelEncodedTrain.csv',index=False)
val_data.to_csv('data/isic/labelEncodedVal.csv',index=False)