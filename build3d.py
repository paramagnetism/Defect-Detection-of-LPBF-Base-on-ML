# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:18:14 2024

@author: 0210s
"""
import os
import cv2
import pickle
import numpy as np
from mayavi import mlab
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# local lib
from OT_mean import OT_mean
from MPM_mean import MPM_mean
from findrect import Pointlist

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


class Slice:
    def __init__(self, OT_picname, MPM_root, mode, rectnum = 22):
        OT = OT_mean(OT_picname, rectnum = rectnum)
        OT.divpart = 1
        OT.FullProcess()
        OT.cal_mean(show = False) 
        OT.matching_strip()
        self.OT_strip = OT.strip_color
        
        # if tiff image
        if MPM_root[-4:] == '.tif':
            MPM = MPM_mean(MPM_root, rectnum = rectnum)
            MPM.divpart = 1
            MPM.FullProcess()
            MPM.cal_mean([], show = False) # 5 / 1
        
        # if is root name
        else:
            for root, dirs, files in os.walk(MPM_root):
                files.sort()
                filenames = [os.path.join(root, filename) for filename in files]
                
                # Use modol = on as default
                point_list = Pointlist(filenames[2], filenames[3], 
                                       filenames[mode], params = 'anticlock')
                
                gray = point_list.imshow()
                MPM = MPM_mean(gray, rectnum = rectnum)
                MPM.FullProcess()
                MPM.cal_mean(point_list, show = False)
                
        try:  # if the MPM figure is corrupted
            MPM.matching_strip()
            self.MPM_strip = MPM.strip_color
            self.region = MPM.erosion.astype(np.float32)
            self.ctrs = MPM.ctrs
            
        except IndexError:
            print('MPM image deprecated in '+MPM_root)
            self.MPM_strip = 0
            # need to resize the OT image to MPM scale
            self.region = cv2.resize(OT.erosion, (2500,2500)).astype(np.float32)
            self.ctrs = OT.ctrs
            # self.region = np.zeros((2500,2500))
            
class Model3D:
    def __init__(self, roots, Calib = 6, Param = 'IntOn', badmpm = 3):
        self.__calib = Calib
        if 'Int' in Param:
            self.__OT_type = 'Int'
        elif 'Max' in Param:
            self.__OT_type = 'Max'
        else:
            self.__OT_type = ''
        if 'On' in Param:
            self.__MPM_type = 1
        elif 'Off' in Param:
            self.__MPM_type = 0
        else:
            self.__MPM_type = ''
        
        # OT is in 2 Directories
        if type(roots) == str and os.path.exists(roots):
            self.Slices = np.load(roots)
        else:
            self.OT_root = os.path.join(roots[0], self.__OT_type.upper())
            # MPM are all in one directory
            self.MPM_root = roots[1]
            # RAM slices
            self.__buffer = {}
            self.badmpm = badmpm
            
            models = ReadModels(dirct = 'models/')
            # Use MPM + OT model as default, OT only when MPM deprecated
            self.model = models[0]['model']
            self.OTmodel = models[1]['model']
        
    # Given index name, update value in self.__buffer    
    def __loadfiles(self, Range):
        [start_idx, end_idx] = Range
        for i in range(start_idx, end_idx):
            if i not in self.__buffer.keys():
                # Can change MPM to pic
                numb = (i + 1) * 3
                ten = numb % 100
                hund = int((numb - ten)/100)
                MPM_name = '00'+ str(hund)+ '_' + str(ten).zfill(2) + '0'
                OT_name = 'SI246120231031190932_' + str(i)+'_' + MPM_name +'_' \
                    + self.__OT_type + '_32F.tif'
                OT_name = os.path.join(self.OT_root, OT_name)
                
                # if MPM is from excel files
                MPM_name1 = os.path.join(self.MPM_root, MPM_name)
                if os.path.exists(MPM_name1):
                    MPM_name = MPM_name1 
                else:
                    self.__MPM_type = 'on' if self.__MPM_type == 1 else 'off'
                    MPM_name = 'Vaildation print 31102023_SI246120231031190932_' \
                        + MPM_name + '_00.mpm_ma_' + self.__MPM_type + 'axis.tif'
                    MPM_name = os.path.join(self.MPM_root, MPM_name)
                self.__buffer.update({i:Slice(OT_name, MPM_name, self.__MPM_type)})
        # delete keys if unused        
        for key in list(self.__buffer.keys()):
            # remain some for deprecated 
            if key < start_idx - self.badmpm or key > end_idx + self.badmpm:
                del self.__buffer[key]
                
    def calibMPM(self, Range, idxmask, savename = '../LargeFiles/demo'):
        [start_idx, end_idx] = Range
        # Calib MPM
        print('Reading data')
        indexs = list(range(start_idx, end_idx))  # All index
        self.__loadfiles([start_idx, start_idx + self.__calib]) # Initialize
        # MASK for floodfill
        MASK = np.zeros((2502, 2502)).astype(np.uint8)
        # Memorize the deprecated MPM
        self.deprecated_mpm = []
        self.Slices = []
        for i in tqdm(indexs):  # for all mpm mean
            depth = []            
            # if MPM image is corrupted
            if self.__buffer[i].MPM_strip == 0:
                self.deprecated_mpm.append(i)
                for j, otstrip in enumerate(self.__buffer[i].OT_strip):
                    OTmean = otstrip['OT_mean']
                    depth.append(self.OTmodel.predict(np.array([[OTmean, 370, 1300]])))
            else: 
                last_depcount = 0
                while True:
                    deprcount = 0
                    # Load the file
                    nearest = sorted(indexs, key = lambda n: abs(i - n))[:self.__calib
                                                                        + deprcount]

                    self.__loadfiles([min(nearest), max(nearest) + self.badmpm])
                    # Check if there's deprecated MPM
                    for slices in self.__buffer.values():
                        if slices.MPM_strip == 0 :
                            deprcount += 1
                    # If there's no extra ones
                    if last_depcount == deprcount:
                        break
                    else:  # Find another MPM
                        last_depcount = deprcount 
                
                # caculate mean mpm for nearest figure
                slices = [self.__buffer[j].MPM_strip for j in nearest if self.__buffer[j].MPM_strip != 0]
                Sum = [0]*len(slices[0])
                for Slic in slices:
                    for j in range(len(Slic)): #22
                        Sum[j]+=Slic[j]['MPM_mean']
                Std = [j/len(slices) for j in Sum]

                # normalize mpm and predict depth
                for j, mpmstrip in enumerate(self.__buffer[i].MPM_strip):
                    OTmean = self.__buffer[i].OT_strip[j]['OT_mean']
                    calib = mpmstrip['MPM_mean']/Std[j]/1300*10000
                    depth.append(self.model.predict(np.array([[OTmean, calib, 370, 1300]])))
            
            # paint the depth on map
            for j, ctr in enumerate(self.__buffer[i].ctrs):
                color = depth[j] if j in idxmask else 0.0
                cv2.floodFill(self.__buffer[i].region, MASK, ctr, color, 
                              (1,1,1),(1,1,1), cv2.FLOODFILL_FIXED_RANGE)
            self.Slices.append(self.__buffer[i].region)
        
        # Transpose for 3D drawing
        self.Slices = np.transpose(np.array(self.Slices),(1,2,0))
        
        # repair the deprecated, Can have sth to do here
        for i in self.deprecated_mpm:
            idx = i+1 if (i<end_idx-2 and i+1 not in self.deprecated_mpm) else i-1
            self.Slices[:,:,i-start_idx][self.Slices[:,:, idx-start_idx]==0] = 0
        np.save(savename + '_original', self.Slices)
        # update depth 
        for i in range(self.Slices.shape[2]-1):
            upper = self.Slices[:,:,-i-1]-30
            self.Slices[:,:,-i-2] = np.maximum(self.Slices[:,:,-i-2], upper)
        np.save(savename + '_aftr', self.Slices)

    def Model_3D(self):
        print('Rendering…………')
        pixel_length_cm = 25 / 2500  # cm
        pixel_depth_cm = 30e-4  # cm
        # Resample the slices to represent the exact length and Create mesh grid
        xx, yy, zz = np.meshgrid(np.arange(0, self.Slices.shape[0]) * pixel_length_cm,
                                 np.arange(0, self.Slices.shape[1]) * pixel_length_cm,
                                 np.arange(0, self.Slices.shape[2]) * pixel_depth_cm,
                                 indexing='ij')
        # Visualize using Mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, self.Slices)
        vol = mlab.pipeline.iso_surface(src)
        vol.actor.property.opacity = 0.5
        return src, vol
        
if __name__ == '__main__':
    model3Dname = '../LargeFiles/demo_aftr.npy'
    if os.path.exists(model3Dname):
        obj = Model3D(model3Dname)
    else:
        obj = Model3D(['D:\OT', 'D:\MPMTIFF'])
        obj.calibMPM(Range = [0,200],
                     idxmask = [i for i in range(22)],
                     savename = '../LargeFiles/demo')
    src, vol = obj.Model_3D()
    mlab.show()