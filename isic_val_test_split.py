# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:55:29 2020

@author: prnvb
"""

import pandas as pd
from sklearn.model_selection import train_test_split

val_gt = pd.read_csv('data/isic/val_gt.csv')

isic_test, isic_val = train_test_split(val_gt,test_size=0.2)

isic_test.to_csv('data/isic/isic_test_gt.csv',index=False)
isic_val.to_csv('data/isic/isic_val_gt.csv',index=False)