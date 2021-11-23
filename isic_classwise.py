# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:42:57 2020

@author: prnvb
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

BASE_DIR = 'data/isic/'

df = pd.read_csv(BASE_DIR+'ISIC_2019_Training_GroundTruth.csv')
df['X']=df['image']+'.jpg'
df['Y'] = df.iloc[:,1:10].idxmax(axis=1)
df['Y_LE'] = 0

mapp = {}
cols = list(df.columns)[1:9]
for i in range(len(cols)):
    mapp[cols[i]] = i
for i in range(len(df)):
    df['Y_LE'][i] = mapp[df['Y'][i]]


train_df,val_df = train_test_split(df,test_size=0.2,random_state=10)

train_df.to_csv(BASE_DIR+'train_gt.csv',index=False)
val_df.to_csv(BASE_DIR+'val_gt.csv',index=False)

class_wise = {}
for i in range(1,9):
    class_wise[i-1] = list(df[ df[df.columns[i]] == 1 ]['image'])

joblib.dump(class_wise,'data/isic/class_wise.pkl')


train_class_wise = {}
for i in range(1,9):
    train_class_wise[i-1] = list(train_df[train_df[train_df.columns[i]]==1]['image'])
joblib.dump(class_wise,'data/isic/train_class_wise.pkl')


val_class_wise = {}
for i in range(1,9):
    val_class_wise[i-1] = list(val_df[val_df[val_df.columns[i]]==1]['image'])
joblib.dump(class_wise,'data/isic/val_class_wise.pkl')


class_map = {}
i=0
for c in list(df.columns)[1:9]:
    class_map[i]=c
    i+=1

class_weights = {}
for c in class_wise.keys():
    class_weights[class_map[c]] = (1-len(class_wise[c])/len(df))*10

joblib.dump(class_weights,'data/isic/class_weights.pkl')

int_class_weights = {}
for c in class_wise.keys():
    int_class_weights[c] = (1-len(class_wise[c])/len(df))*10



#-----------------------------------------------------------



gt = pd.read_csv('data/isic/ISIC_2019_Training_GroundTruth.csv')

class_wise = {}
for i in range(1,9):
    class_wise[i-1] = list(gt[ gt[gt.columns[i]] == 1 ]['image'])

joblib.dump(class_wise,'data/isic/class_wise.pkl')



data = []
for i in range(len(gt)):
    img = gt.iloc[i,0]
    label = np.argmax(gt.iloc[i,1:9])
    data.append([img,label])

t = pd.DataFrame(data,columns=['image','label'])
t.to_csv('data/isic/labelEncoded.csv',index=False)


mapp = {}
cols = list(gt.columns)[1:9]
for i in range(len(cols)):
    mapp[cols[i]] = i

gt['Y'] = gt.iloc[:,1:9].idxmax(axis=1)

for i in range(len(gt)):
    gt['Y'][i] = mapp[gt['Y'][i]]
