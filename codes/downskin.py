# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:25:58 2024

@author: Admin
"""
from poly_regress import *
from sklearn.linear_model import LinearRegression
import numpy as np

import os
import pickle

def ReadModels(dirct = 'models/'):
    models = []
    for root, dirs, files in os.walk(dirct):
        filenames = [os.path.join(root, filename) for filename in files]
        filenames.sort(key = lambda x: -len(x))
    for filename in filenames:
        with open(filename,"rb") as f:
            loaded = pickle.load(f)
        models.append(loaded)
    return models

model = ReadModels()[0]['model']
'''
Labels = ['OT_Int (0-255)',         # 0
          'OT_Max (0-255)',         # 1
          'MPM_on (0-255)',         # 2    
          'MPM_off (0-255)',        # 3
          'Calib_on',               # 4
          'Calib_off',              # 5
          'Hatch Space (mm)',       # 6
          'Laser Power (W)',        # 7
          'Scanning Speed (mm/s)']  # 8
labelid = [0, 5, 7, 8]
output = ['D_mean (µm)', 'W_mean (µm)']
trainner = AutoTrain('RESULT.csv', Labels, output, outputid = 0)
X = np.array(trainner.data[[trainner.Labels[i] for i in labelid]])
trainner = AutoTrain('RESULT.csv', Labels, output, outputid = 0)
X = np.array(trainner.data[[trainner.Labels[i] for i in labelid]])
Y = trainner.y
y = model.predict(X)
'''

# Totally unrelated
Labels = ['OT_Int (0-255)',         # 0
          'OT_Max (0-255)',         # 1
          'Calib_on',               # 2
          'Calib_on',               # 3
          'Laser Power (W)',        # 4
          'Scanning Speed (mm/s)']  # 5
output = ['D_mean (µm)']
trainner = AutoTrain('E:/27 OT/MEAN.csv', Labels, output, outputid = 0)
labelid = [4, 5]
X = np.array(trainner.data[[trainner.Labels[i] for i in labelid]])
Y = trainner.y

model = LinearRegression().fit(X, Y)
y = model.predict(X)

from sklearn.metrics import r2_score
r2 = r2_score(Y, y)

with open('models/Dskin.pkl',"wb") as f:
    pickle.dump(model, f)