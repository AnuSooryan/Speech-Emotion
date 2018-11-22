# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 11:05:49 2018

@author: Anu
"""

import os
import numpy as np
import librosa
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

path = os.getcwd()

files = os.listdir(path + '/RawData/')

lb = LabelEncoder()

emotions = []
for i in files:
    if i[6:-16] == '01' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_neutral')
    elif i[6:-16] == '01' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_neutral')
    elif i[6:-16] == '02' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_calm')
    elif i[6:-16] == '02' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_calm')
    elif i[6:-16] == '03' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_happy')
    elif i[6:-16] == '03' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_happy')
    elif i[6:-16] == '04' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_sad')
    elif i[6:-16] == '04' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_sad')
    elif i[6:-16] == '05' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_angry')
    elif i[6:-16] == '05' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_angry')
    elif i[6:-16] == '06' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_fearful')
    elif i[6:-16] == '06' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_fearful')
    elif i[6:-16] == '07' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_disgust')
    elif i[6:-16] == '07' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_disgust')
    elif i[6:-16] == '08' and int(i[18:-4]) % 2 == 0:
        emotions.append('female_surprised')
    elif i[6:-16] == '08' and int(i[18:-4]) % 2 == 1:
        emotions.append('male_surprised')
    
feature_list = []
for indx, i in enumerate(files):
    feature_dict = {}
    y, sr = librosa.load('RawData/'+ i, res_type='kaiser_fast',
                         duration=2.5, sr=22050*2, offset=0.5)
    sr = np.array(sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=0) #axis =0 for 1D
    feature_dict['features'] = mfcc
    feature_dict['label'] = emotions[indx]
    feature_list.append(feature_dict)# list of dict 
    
df = pd.DataFrame(feature_list)
features = pd.DataFrame(df['features'].values.tolist())
data = pd.concat([features, df['label']], axis=1)
data = data.fillna(0)# replace NaN with 0
X = data.drop(['label'], axis=1)
lab = data['label']
unique_lab = list(lab.unique())
Y = lb.fit_transform(lab)
indexes = np.unique(Y, return_index=True)[1]
unique_Y = [Y[index] for index in sorted(indexes)]
finalY = dict(zip(unique_Y, unique_lab))#assign labels to the numerical indexes
joblib.dump(finalY, 'label.pkl')# dictionary of labels .eg.0= male neutral

clf = KNeighborsClassifier()
clf.fit(X, Y)

joblib.dump(clf, 'model.pkl')

