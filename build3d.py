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
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# local lib
from OT_mean import OT_mean
from MPM_mean import MPM_mean
from findrect import Pointlist

class Model3D:
    def __init__(self, roots, Calib = 6, Param = 'IntOn', badmpm = 2):
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
            self._ReadModels(dirct = 'models/')
            # Use MPM + OT model as default, OT only when MPM deprecated
            
            
    def _ReadModels(self, dirct = 'models/'):
        models = []
        for root, dirs, files in os.walk(dirct):
            filenames = [os.path.join(root, filename) for filename in files]
            filenames.sort(key = lambda x: -len(x))
        for filename in filenames:
            with open(filename,"rb") as f:
                loaded = pickle.load(f)
            models.append(loaded)
        self.model = models[0]['model']
        self.OTmodel = models[1]['model']


    class _Slice:
        def __init__(self, OT_picname, MPM_root, mode, rectnum = 22, useOT = False):
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
                if useOT:
                    raise IndexError('In consistent deprecated detected')
                MPM.matching_strip()
                
            except IndexError:
                print('\n'+MPM_root+' deprecated, use OT instead')
                self.MPM_strip = 0
                # need to resize the OT image to MPM scale
                self.region = cv2.resize(OT.erosion, (2500,2500)).astype(np.float32)
                self.ctrs = OT.ctrs
                # self.region = np.zeros((2500,2500))
                
            else:
                self.MPM_strip = MPM.strip_color
                self.region = MPM.erosion.astype(np.float32)
                self.ctrs = MPM.ctrs
    
    def _idx2name(self, i):
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
        return OT_name, MPM_name
    
    # Given index name, update value in self.__buffer    
    def __loadfiles(self, Range):
        [start_idx, end_idx] = Range
        for i in range(start_idx, end_idx):
            if i not in self.__buffer.keys():
                OT_name, MPM_name = self._idx2name(i)
                self.__buffer.update({i:self._Slice(OT_name, MPM_name, self.__MPM_type)})
                # print('read index'+str(i))
                
        # delete keys if unused        
        for key in list(self.__buffer.keys()):
            # remain some for deprecated 
            if key < start_idx - self.badmpm or key > end_idx + self.badmpm:
                del self.__buffer[key]
                # print('delete index'+str(key))
                
    def build(self, Range, idxmask, savename = 'Files/demo'):
        [start_idx, end_idx] = Range
        # Calib MPM
        print('Reading data')
        indexs = list(range(start_idx, end_idx))  # All index
        self.__loadfiles([start_idx, start_idx + self.__calib]) # Initialize
        # Memorize the deprecated MPM
        self.deprecated_mpm = []
        self.Slices = []
        self.depths = []
        for i in tqdm(indexs):  # for all mpm mean
            depth = []
            # Check if MPM image is corrupted
            if self.__buffer[i].MPM_strip == 0 or \
                (i > start_idx and self.__CheskMPM(i, maxdis = 100, show = True)):
                self.deprecated_mpm.append(i)
                for j, otstrip in enumerate(self.__buffer[i].OT_strip):
                    OTmean = otstrip['OT_mean']
                    depth.append(self.OTmodel.predict(np.array([[OTmean, 370, 1300]])))
            else:  # calculate the depth
                last_depcount = 0
                while True:
                    deprcount = 0
                    # Load the file
                    nearest = sorted(indexs, key = lambda n: abs(i - n))[:self.__calib
                                                                        + deprcount]

                    self.__loadfiles([min(nearest), max(nearest) + self.badmpm + 1])
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
                slices = [self.__buffer[j].MPM_strip for j in nearest \
                          if self.__buffer[j].MPM_strip != 0]
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
            
            self.depths.append(depth)
            
            # paint the depth on map
            for j in idxmask:
                # !! This mask should set to np.zeros every time calling the function
                cv2.floodFill(self.__buffer[i].region, np.zeros((2502, 2502)).astype(np.uint8), 
                              self.__buffer[i].ctrs[j], depth[j], 
                              1, 1, cv2.FLOODFILL_FIXED_RANGE)
            
            # remove noise from floodfill and unfilled region
            self.__buffer[i].region[self.__buffer[i].region > max(depth)+1] = 0
            self.__buffer[i].region[self.__buffer[i].region < min(depth)-1] = 0
            self.Slices.append(self.__buffer[i].region)
        
        # Transpose for 3D drawing
        self.Slices = np.transpose(np.array(self.Slices),(1,2,0))
        
        # repair the deprecated, Can have sth to do here
        for i in self.deprecated_mpm:
            idx = i+1 if (i<end_idx-2 and i+1 not in self.deprecated_mpm) else i-1
            self.Slices[:,:,i-start_idx][self.Slices[:,:, idx-start_idx]==0] = 0
        # np.save(savename + '_original', self.Slices)  # Too large to save
        self.__Save01(Savename = savename + '_original')
        np.save(savename + '_depths', np.array(self.depths))
        
        # update depth
        for i in range(self.Slices.shape[2]-1):
            upper = self.Slices[:,:,-i-1]-30
            self.Slices[:,:,-i-2] = np.maximum(self.Slices[:,:,-i-2], upper)
        # np.save(savename + '_aftr', self.Slices)
        self.__Save01(Savename = savename + '_aftr')
        
    # To avoid inconsistent cube recognition
    def __CheskMPM(self, i, maxdis = 100,  show = False):
        for j, ctr in enumerate(self.__buffer[i].ctrs):
            dis = np.sqrt(np.sum((ctr - self.__buffer[i-1].ctrs[j])**2))
            if dis > maxdis:
                print("\nIn consistent deprecated detected on "+str(i)+" : " \
                      +str(j)+'distance ='+str(dis))
                    
                if show:
                    img1 = cv2.cvtColor(self.__buffer[i-1].region, cv2.COLOR_GRAY2BGR)
                    img2 = cv2.cvtColor(self.__buffer[i].region, cv2.COLOR_GRAY2BGR)
                    for count, center in enumerate(self.__buffer[i-1].ctrs):
                        cv2.putText(img1, str(count), center,
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    2, (0,255,0), 2)
                    for count, center in enumerate(self.__buffer[i].ctrs):
                        cv2.putText(img2, str(count), center,
                                    cv2.FONT_HERSHEY_COMPLEX, 
                                    2, (0,255,0), 2)
                    img1 = cv2.resize(img1,(img1.shape[1]>>1, img1.shape[0]>>1))
                    img2 = cv2.resize(img2,(img2.shape[1]>>1, img2.shape[0]>>1))
                    cv2.imshow("img"+str(i-1),img1)
                    cv2.imshow("img"+str(i),img2)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    
                OT_name, MPM_name = self._idx2name(i)
                # Replace the corrupted with OT
                self.__buffer.update({i:self._Slice(OT_name, MPM_name, self.__MPM_type, useOT = True)})
                return True
        return False
    
    def __Save01(self, Savename):
        Slices = np.zeros(self.Slices.shape).astype(np.uint8)
        Slices[self.Slices>0] = 255
        np.save(Savename, Slices)
    
    def ShowSlice(self, i):
        #cv2.imshow('test', obj.Slices[:,:,0].astype(np.uint8))
        cv2.imshow('test',cv2.resize(self.Slices[:,:,i].astype(np.uint8),(1000,1000)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def ShowVideo(self):
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        for i in range(self.Slices.shape[2]):
            slice_resized = cv2.resize(self.Slices[:,:,i], (500, 500))
            cv2.putText(slice_resized, str(i), (20, 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255), 1)
            cv2.imshow('test', slice_resized)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
              
    def Model_3D_surface(self):
        print('Rendering Surface…………')
        pixel_length_cm = 1e-2  # cm
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
    
    # Success plot
    def Model_3D_dots(self, originalPath, aftrPath):
        print('Rendering dot clouds…………')
        original = np.load(originalPath)
        aftr = np.load(aftrPath)
        x, y, z = np.indices(original.shape)
        
        x_real = x * 0.01
        y_real = y * 0.01
        z_real = z * 0.003
        
        indices_original = np.argwhere(original > 0)
        np.random.shuffle(indices_original)
        indices_original = indices_original[:50000]
        
        indices_aftr_only = np.argwhere(aftr > original)
        np.random.shuffle(indices_aftr_only)
        indices_aftr_only = indices_aftr_only[:10000]
        
        fig = mlab.figure(size=(800,800))
        mlab.points3d(x_real[indices_original[:, 0], indices_original[:, 1], indices_original[:, 2]],
              y_real[indices_original[:, 0], indices_original[:, 1], indices_original[:, 2]],
              z_real[indices_original[:, 0], indices_original[:, 1], indices_original[:, 2]],
              color=(0, 1, 0), mode='point', scale_factor=1)

        mlab.points3d(x_real[indices_aftr_only[:, 0], indices_aftr_only[:, 1], indices_aftr_only[:, 2]],
              y_real[indices_aftr_only[:, 0], indices_aftr_only[:, 1], indices_aftr_only[:, 2]],
              z_real[indices_aftr_only[:, 0], indices_aftr_only[:, 1], indices_aftr_only[:, 2]],
              color=(1, 0, 0), mode='point', scale_factor=1)

        mlab.show()
        return fig
        
    # This function is a faliure    
    def Model_2D(self, loadname):
        print('Rendering…………')
        
        original = np.load(loadname + '_original.npy')
        aftr = np.load(loadname + '_aftr.npy')
        # Filter aftr where original is 0
        # diff = np.where(original == 0, aftr, 0)
        x_scale = y_scale = 0.04
        z_scale = 0.003
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # indices = np.nonzero(diff)
        
        # Plot non-zero points
        for i in tqdm(range(aftr.shape[0]>>2)):
            for j in range(aftr.shape[1]>>2):
                for k in range(aftr.shape[2]):
                    if original[i<<2, j<<2, k] < aftr[i<<2, j<<2, k]:
                        ax.scatter(i * x_scale, j * y_scale, k * z_scale, color='red', alpha = 1)
                    elif original[i, j, k] > 0:
                        ax.scatter(i * x_scale, j * y_scale, k * z_scale, color='gray', alpha = 1)
                
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        
        plt.title('Region Representation in 3D')
        plt.show()
    
if __name__ == '__main__':
    model3Dname = 'Files/demo'
    openname = model3Dname + '_original.npy'
    if os.path.exists(openname):
        obj = Model3D(openname)
    else:
        obj = Model3D(['E:\OT', 'E:\MPMTIFF'])
        obj.build(Range = [0,200],
                  idxmask = [i for i in range(22)],
                  savename = model3Dname)

    fig = obj.Model_3D_dots(model3Dname+ '_original.npy',
                        model3Dname+ '_aftr.npy')
    
    # obj.ShowSlice(162)
    # obj.ShowVideo()
    
    # src, vol = obj.Model_3D_surface()
    # mlab.show()