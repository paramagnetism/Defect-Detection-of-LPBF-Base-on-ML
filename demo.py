# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:25:58 2024

@author: Admin
"""
from build3d import ReadModels
from poly_regress import *

model = ReadModels()[0]['model']
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
Y = trainner.y
y = model.predict(X)
models = ReadModels()[0]
