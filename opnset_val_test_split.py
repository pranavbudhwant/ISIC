# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:20:23 2020

@author: prnvb
"""

import pandas as pd
from sklearn.model_selection import train_test_split

openset_data = pd.read_csv('data/openset_data.csv')

openset_test, openset_val = train_test_split(openset_data,test_size=0.2,random_state=10)

openset_test['Y_LE'].value_counts()
openset_val['Y_LE'].value_counts()

openset_test.to_csv('data/openset_test.csv',index=False)
openset_val.to_csv('data/openset_val.csv',index=False)

