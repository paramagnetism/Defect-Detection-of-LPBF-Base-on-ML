# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:48:41 2024

@author: Admin
"""
import os
import cv2
import pickle
import numpy as np
from OT_mean import OT_mean
from MPM_mean import MPM_mean

def ReadModels(dirct = 'models/'):
    models = []
    for root, dirs, files in os.walk(dirct):
        filenames = [os.path.join(root, filename) for filename in files]
        
        # Sorted models by length of filenames
        filenames.sort(key = lambda x: -len(x))
        
    for filename in filenames:
        with open(filename,"rb") as f:
            loaded = pickle.load(f)
        models.append(loaded)
    model = models[0]['model']
    OTmodel = models[1]['model']
    Downskin = models[3].predict(np.array([[340,1150]]))[0]
    print("Applied all downskin depth: "+str(Downskin))
    return model, OTmodel, Downskin

#OTdown = cv2.imread('E:/OT/INT/SI246120231031190932_249_007_500_Int_32F.tif', cv2.IMREAD_GRAYSCALE)
#OTup = cv2.imread('E:/OT/INT/SI246120231031190932_250_007_530_Int_32F.tif', cv2.IMREAD_GRAYSCALE)
MPMup = MPM_mean('E:/MPMTIFF/Vaildation print 31102023_SI246120231031190932_007_530_00.mpm_ma_offaxis.tif', rectnum = 12)
MPMdown = MPM_mean('E:/MPMTIFF/Vaildation print 31102023_SI246120231031190932_007_500_00.mpm_ma_offaxis.tif', rectnum = 12)
MPMup.FullProcess(close = 15)
MPMdown.FullProcess(close = 15)

erosion = cv2.morphologyEx(MPMup.erosion, cv2.MORPH_OPEN, (3,3))

mpm = MPMup.src.copy()
mpm[erosion > MPMdown.erosion] = 0
MPM = MPMup.src.copy()
MPM[MPM == mpm] = 0
#OTup[OTup < OTdown+20] = 0
#MPMup[MPMup < MPMdown+20] = 0
cv2.imwrite('mpm.bmp', MPM)

MPM = MPM_mean('mpm.bmp', rectnum = 12)
MPM.FullProcess(close = 15)
MPM.cal_mean([], show = True)
test1 = MPM.mean
  
OTup = OT_mean('E:/OT/INT/SI246120231031190932_250_007_530_Int_32F.tif', rectnum = 12)
img = OTup.src.copy()
img[MPM.erosion == 0] = 0
OTup.Set(img)
OTup.FullProcess()
OTup.cal_mean(show = True)
test2 = OTup.mean

save = OTup.src.copy()
save[MPM.erosion == 0] = 0
cv2.imwrite("ot.bmp", save)

model, OTmodel, Downskin = ReadModels()
rst1 = []
rst2 = []
for i in range(12):
    rst1.append(model.predict(np.array([[test2[i], 1/1150, 340, 1150]]))[0])
    rst2.append(OTmodel.predict(np.array([[test2[i], 340,1150]]))[0]+Downskin)
    