'''
Created on 07-Nov-2018

@author: anu
'''
import os
import numpy as np
import librosa
import pandas as pd
from sklearn.externals import joblib

path = os.getcwd() + '/SpeechEmotion/SpeechEmotion'

def Emotion(audio):
    model = joblib.load(path + '/model.pkl')
    print(model)
    labels = joblib.load(path + '/label.pkl')
    print(labels)
    y, sr = librosa.load(audio, res_type='kaiser_fast', 
                         duration=2.5, sr=22050*2, 
                         offset=0.5)
    sr = np.array(sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=0)
    feature_list = []
    feature_dict = {}
    feature_dict['features'] = mfcc
    feature_list.append(feature_dict)
    df = pd.DataFrame(feature_list)
    features = pd.DataFrame(df['features'].values.tolist())
    prediction_array = model.predict(features)
    prediction = labels[prediction_array[0]]
    result = {}
    result['Gender'] = prediction.split('_')[0].title()
    result['Emotion'] = prediction.split('_')[1].title()
    return result