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

# local lib
from OT_mean import OT_mean
from MPM_mean import MPM_mean
from findrect import Pointlist, FindContour

class Model3D:
    def __init__(self, roots, prefix,
                 Calib = 6, 
                 Param = 'IntOn', 
                 badmpm = 2,
                 # Laser Power, Scanning Speed = 370, 1300 as default
                 Settings = [340, 1150]): # 
        # Calibrated number should be larger than 5 
        self.__calib = max(5, Calib)
        # Need to adjuct
        self.expected_rect = [46]*207 + [134]*33 + [24] * 10 + [0] + [12] * 64
        [self.OT_prefix, self.MPM_prefix] = prefix
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
            # If exist, load the Slices directly
            self.Slices = np.load(roots)
        else:
            self.OT_root = os.path.join(roots[0], self.__OT_type.upper())
            # MPM are all in one directory
            self.MPM_root = roots[1]
            # RAM slices
            self.__buffer = {}
            self.badmpm = badmpm
            # Use MPM + OT model as default, OT only when MPM deprecated
            self._Settings = Settings
            self._ReadModels(dirct = 'models/')
    
      
    def _ReadModels(self, dirct = 'models/'):
        models = []
        for root, dirs, files in os.walk(dirct+'modelsImageD/'):
            filenames = [os.path.join(root, filename) for filename in files]
            
            # Sorted models by length of filenames
            filenames.sort(key = lambda x: -len(x))
            
        for filename in filenames:
            with open(filename,"rb") as f:
                loaded = pickle.load(f)
            models.append(loaded)
        self.model = models[0]['model']
        self.OTmodel = models[1]['model']        
        
        for root, dirs, files in os.walk(dirct+'modelsImageW/'):
            filenames = [os.path.join(root, filename) for filename in files]
            
            # Sorted models by length of filenames
            filenames.sort(key = lambda x: -len(x))
            
        for filename in filenames:
            with open(filename,"rb") as f:
                loaded = pickle.load(f)
            models.append(loaded)
        self.modelW = models[0]['model']
        self.OTmodelW = models[1]['model']
        
        with open(dirct+'Dskin.pkl',"rb") as f:
            loaded = pickle.load(f)                            
        self.Downskin = loaded.predict(np.array([self._Settings]))[0]
        print("Applied all downskin depth: "+str(self.Downskin))
        
        
    class _Slice:
        def __init__(self, OT_picname : str, MPM_root : str,
                     mode : int, rectnum: int, general = True):
            if general:
                MPM = FindContour(MPM_root, rectnum = rectnum)
                OT = FindContour(OT_picname, rectnum = rectnum)
                MPM.FullProcess(show = False)
                
            else:
                OT = OT_mean(OT_picname, rectnum = rectnum)
                OT.sorting_slope = 10
                
                # if tiff image
                if MPM_root[-4:] == '.tif':
                    MPM = MPM_mean(MPM_root, rectnum = rectnum)
                    MPM.sorting_slope = 10
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
                    
            self.rectnum = rectnum
            self.region = MPM.erosion.astype(np.float32)
            # Calculate OT + MPM          
            if len(MPM.mean) == rectnum and min(MPM.mean):
                OT.erosion = MPM.erosion
                if general:
                    OT.get_contours()
                else:
                    OT.get_rotate()
                    OT.find_inner_rects()
                    
                OT.cal_mean()
                self.OT_strip = OT.mean
                self.MPM_strip = MPM.mean
                self.ctrs = MPM.ctrs
                self.contours = MPM.contours
            else:
                print('\n'+ MPM_root +' deprecated, use OT instead')
                self.MPM_strip = 0
                self.ctrs = []
                # Caculate only with OT
                self.OT = OT
                self.OT.threshold(3)
                self.OT.morphoperation(15)
        
        def reconstruct(self, slice_1):
            '''
            Use 2 Silces before to generate ctrs, contours and region

            Parameters
            ----------
            slice_1 : Slice
                The -1 Slice.
            '''
            # Cut the region out for OT to caculate
            self.OT.erosion[slice_1.region == 0] = 0
            self.region[self.region == 0] = self.OT.erosion[self.region == 0]
            self.OT.erosion = self.region.astype(np.uint8)
            self.OT.get_contours(show = False)
            self.OT.cal_mean()
            
            try:
                assert len(self.OT.mean) == self.rectnum
            except AssertionError:
                self.OT.get_contours(show = True)
                self.OT.cal_mean()
                print(len(self.OT.contours))
                print(len(self.OT.mean))
                print(self.rectnum)
                
            self.OT_strip = self.OT.mean
            self.ctrs = self.OT.ctrs
            self.contours = self.OT.contours
            self.OT = True
        
        def recenter(self, centers):
            '''
            To relocate all examined centers according to nearest distance
            ------
            Parameters:
                centers: (list of np array) centers from last layer
                
            Returns:
                None
            '''
            if centers:
                # print(len(centers))
                idx = [sorted([_ for _ in range(len(centers))], 
                                 key = lambda x: np.linalg.norm(self.ctrs[x]
                                - ctr))[0] for ctr in centers]
                
                # print(len(self.contours))
                self.ctrs = [self.ctrs[i] for i in idx]
                self.contours = [self.contours[i] for i in idx]
                self.OT_strip =  [self.OT_strip[i] for i in idx]
                if self.MPM_strip != 0:
                    self.MPM_strip =  [self.MPM_strip[i] for i in idx]
                    

    def _idx2name(self, i):
        '''
        To trasnlate loading index to 
        ------
        Parameters:
            centers: (list of np array) centers from last layer
            
        Returns:
            None
        '''
        numb = (i + 1) * 3
        ten = numb % 100
        hund = int((numb - ten)/100)
        MPM_name = '00'+ str(hund)+ '_' + str(ten).zfill(2) + '0'
        OT_name = self.OT_prefix + str(i)+'_' + MPM_name +'_' \
            + self.__OT_type + '_32F.tif'

        OT_name = os.path.join(self.OT_root, OT_name)
        # if MPM is from excel files
        MPM_name1 = os.path.join(self.MPM_root, MPM_name)
        if os.path.exists(MPM_name1):
            MPM_name = MPM_name1 
        else:
            self.__MPM_type = 'on' if self.__MPM_type == 1 else 'off'

            MPM_name = self.MPM_prefix + MPM_name + '_00.mpm_ma_' \
            + self.__MPM_type + 'axis.tif'
                
            MPM_name = os.path.join(self.MPM_root, MPM_name)
        
        return OT_name, MPM_name
    
    # Given index name, update value in self.__buffer    
    def __loadfiles(self, Range):
        [start_idx, end_idx] = Range
        for i in range(start_idx, end_idx):
            if i not in self.__buffer.keys():
                OT_name, MPM_name = self._idx2name(i)
                self.__buffer.update({i:self._Slice(OT_name, MPM_name, 
                                self.__MPM_type, self.expected_rect[i])})
                
                # print('load' + str(i) + MPM_name)
                # Check if MPM is corrupted
                if i-1 in self.__buffer.keys():
                    if self.__buffer[i].MPM_strip == 0:
                        self.__buffer[i].reconstruct(self.__buffer[i-1])
        
                    # Recenter all the points
                    self.__buffer[i].recenter(self.__buffer[i-1].ctrs)  
                
        # delete keys if unused        
        for key in list(self.__buffer.keys()):
            # remain some for deprecated 
            if key < start_idx - self.badmpm: 
                del self.__buffer[key]
                # print('delete index'+str(key))
                
    def build(self, Range, idxmask, savename = 'Files/demo'):
        '''
        Build slices from images
        --------
        Parameters:
            Range: (int) The 3D NumPy matrix.
            start_point (tuple): The starting point (x, y) of the slice on every layer.
            end_point (tuple): The ending point (x, y) of the slice on every layer.
            
        Returns:
            ndarray: The sliced matrix.
        '''
        [start_idx, end_idx] = Range
        # Calib MPM
        print('Reading data')
        indexs = list(range(start_idx, end_idx))  # All index
        
        # Memorize the deprecated MPM
        self.deprecated_mpm = []
        self.Slices = []
        
        for i in tqdm(indexs):  # for all mpm mean
        # Initialize for each part
            if i == start_idx or self.expected_rect[i] != self.expected_rect[i-1]:
                if self.expected_rect[i] == self.expected_rect[i+1]:
                # Empty buffer for new portion
                    self.__buffer = {}
                    self.__loadfiles([i, i + self.__calib])
                else:
                    OT_name, MPM_name = self._idx2name(i)
                    MPM = cv2.imread(MPM_name, cv2.IMREAD_GRAYSCALE)
                    mpm = MPM.copy()
                    # downskin region
                    mpm[MPM > self.__buffer[i-1].region] = 0 
                    # normal region
                    MPM[MPM==mpm] = 0
                    cv2.imwrite("downskin.bmp", mpm)
                    cv2.imwrite("normal.bmp", MPM)
                    
                
            '''
            Check if MPM image is corrupted, if use function place below:
                i should be larger than start index
            '''
            
            depth = []
            width = []
            if self.__buffer[i].MPM_strip == 0:
                try:
                    assert i > start_idx
                except AssertionError:
                    print("First MPM deprecated!")
                    
                self.deprecated_mpm.append(i)
                for otstrip in self.__buffer[i].OT_strip:
                    depth.append(self.OTmodel.predict(
                        np.array([[otstrip]+self._Settings])))
                    width.append(self.OTmodelW.predict(
                        np.array([[otstrip]+self._Settings]))[0])
            
            else:  # Use MPM and OT together to calculate the depth
                last_depcount = 0
                while True:
                    deprcount = 0
                    # Load the file
                    capable = [j for j in indexs if \
                               self.expected_rect[j] == self.expected_rect[i]]
                    nearest = sorted(capable, key = lambda n: abs(i - \
                                n))[:self.__calib + deprcount  + self.badmpm]
                    # because loadfile don't load the last one
                    self.__loadfiles([min(nearest), max(nearest)+1])

                    # Check if there's deprecated MPM in range
                    for slices in self.__buffer.values():
                        if slices.MPM_strip == 0 :
                            deprcount += 1
                    # If there's no extra ones
                    if last_depcount == deprcount:
                        break
                    else:  # Find another MPM
                        last_depcount = deprcount 
                
                #print('nearest')
                #print(nearest)
                #print('buffer')
                #print(self.__buffer.keys())
                
                # caculate mean mpm for nearest figure
                slices = [self.__buffer[j].MPM_strip for j in nearest \
                          if (self.__buffer[j].MPM_strip != 0 and j != i)]
                Std = np.mean(np.array(slices), axis = 0)
                
                # normalize mpm and predict depth
                for j, mpmstrip in enumerate(self.__buffer[i].MPM_strip):
                    OTmean = self.__buffer[i].OT_strip[j]
                    calib = mpmstrip/Std[j]/self._Settings[1]*10000
                    depth.append(self.model.predict(np.array([[OTmean, calib] + self._Settings])))
                    width.append(self.modelW.predict(np.array([[OTmean, calib] + self._Settings]))[0])
            
            # paint the depth and width on map
            # for j in idxmask:
            for j in [i for i in range(len(depth))]:
                # !! This mask should set to np.zeros every time calling the function
                iters = int(np.floor(width[j]/100))
                if width[j] - iters > 2/3:
                    iters += 1 
                mask = cv2.drawContours(np.zeros((2500,2500)), self.__buffer[i].contours, j, (255))
                mask = cv2.dilate(mask, (3,3), iters)
                if width[j] - iters > 1/3 and width[j] - iters < 2/3:
                    mask = cv2.dilate(mask, (2,2))
                
                self.__buffer[i].region = np.maximum(mask, self.__buffer[i].region).astype(np.uint8)
                
                cv2.floodFill(self.__buffer[i].region, np.zeros((2502, 2502)).astype(np.uint8), 
                              self.__buffer[i].ctrs[j], depth[j], 
                              1, 1, cv2.FLOODFILL_FIXED_RANGE)
            
            # remove noise from floodfill and unfilled region
            self.__buffer[i].region[self.__buffer[i].region > max(depth)+1] = 0
            self.__buffer[i].region[self.__buffer[i].region < min(depth)-1] = 0
            
            # Unfilled aera will remain 255 and then got removed
            self.__buffer[i].region[self.__buffer[i].region == 255] = 0
            self.Slices.append(self.__buffer[i].region)
        
        
        # Release the buffer
        self.buffer = {}
        
        # Transpose for 3D drawing
        self.Slices = np.transpose(np.array(self.Slices),(1,2,0))
            
        # Too large to save painted ones, save unpainted ones instead
        self.__Save01(Savename = savename + '_original')
        # np.save(savename + '_depths', np.array(self.depths))
        
    
    def updateDepth(self, savename, Downskin = True):
        # update depth
        if Downskin:
            self.Slices[self.Slices > 0] += self.Downskin
        
        for i in range(self.Slices.shape[2]-1):
            upper = self.Slices[:,:,-i-1]-30
            self.Slices[:,:,-i-2] = np.maximum(self.Slices[:,:,-i-2], upper)
        # np.save(savename + '_aftr', self.Slices)
        self.__Save01(Savename = savename + '_after')
        
    
    def __Save01(self, Savename):
        Slices = np.zeros(self.Slices.shape).astype(np.uint8)
        Slices[self.Slices>0] = 255
        np.save(Savename, Slices)
        
    
    def ShowSlice(self, i):
        '''
        
        Show the image of Slice index for check
        --------
        Parameters:
            i (int): The index of layer.
        
        '''
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
        '''

        Returns
        -------
            
        # src, vol = obj.Model_3D_surface()
        # mlab.show()

        '''
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
        mlab.show()
        return src, vol
    
    # Success plot
    def Model_3D_dots(self, originalPath):
        print('Rendering dot clouds…………')
        original = np.load(originalPath)
        x, y, z = np.indices(original.shape)
        
        x_real = x * 0.01
        y_real = y * 0.01
        z_real = z * 0.003
        
        indices_original = np.argwhere(original > 0)
        np.random.shuffle(indices_original)
        indices_original = indices_original[:50000]
        
        indices_aftr_only = np.argwhere(self.Slices > original)
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
    
    def CutSlice(self, start, end, originalPath):
        """
        Cut a slice from the 3D matrix vertically based on start and end points.
        
        Parameters:
            matrix (ndarray): The 3D NumPy matrix.
            start_point (tuple): The starting point (x, y) of the slice on every layer.
            end_point (tuple): The ending point (x, y) of the slice on every layer.
            
        Returns:
            ndarray: The sliced matrix.
        """
        original = np.load(originalPath)
        # Extracting start and end points
        if sum(start) < sum(end):
            x1, y1 = start
            x2, y2 = end
        else:
            x1, y1 = end
            x2, y2 = start
            
        MaX, MaY = self.Slices.shape[0], self.Slices.shape[1]
        assert 0 <= x1 and x1 <= MaX-1
        assert 0 <= x2 and x2 <= MaX-1
        assert 0 <= y1 and y1 <= MaY-1
        assert 0 <= y2 and y2 <= MaY-1
        
        # Check the slope
        if x1 == x2: return self.Slices[x1,:,:]
        if y1 == y2: return self.Slices[:,y1,:]
        
        k = (y1-y2)/(x1-x2)
        b = y2 - k*x2
        intersections = [(0, b),
                         (MaX, MaX*k + b),
                         (-b/k, 0),
                         ((MaY-b)/k, MaY)
                         ]
        Intrsct = [pt for pt in intersections if pt[0] <= MaX and \
                    pt[0]>= 0 and pt[1] <= MaY and pt[1]>= 0]
        assert len(Intrsct) == 2
        # Draw start from the smaller X value
        begin, stop = min(Intrsct), max(Intrsct)
        Dx, Dy = stop[0]-begin[0], stop[1]-begin[1]
        dis = (Dx**2 + Dy**2)**0.5
        dx, dy = Dx/dis, Dy/dis
        #print(dx, dy)
        points = [(round(begin[0]+d*dx), round(begin[1]+d*dy))  for d in range(round(dis))]
        Slice = np.flipud(np.array([self.Slices[pts[1], pts[0], :] for pts in points]).T)
        origin = np.flipud(np.array([original[pts[1], pts[0], :] for pts in points]).T)
        Real = cv2.resize(Slice, (round(Slice.shape[1]*10/3), Slice.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
        real = cv2.resize(origin, (round(origin.shape[1]*10/3), origin.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        color = cv2.cvtColor(Real, cv2.COLOR_GRAY2BGR)
        color[Real > real]=(0,255,0)
        cv2.imwrite('Files/slice.png', color)
        cv2.imwrite('Files/origin.png', real)
        cv2.imwrite('Files/after.png', Real)
        return Real
        
if __name__ == '__main__':
    model3Dname = 'Files/demo'
    openname = model3Dname + '_after.npy'
    if os.path.exists(openname):
        obj = Model3D(openname, 
                      
                      ['SI246120231031190932_',
                       'Vaildation print 31102023_SI246120231031190932_'])
                      # ['SI246120230324152157_',
                      # 'MPM-Mold-Figures_SI246120230324152157_'])
    else:
        obj = Model3D(['E:\OT', 'E:\MPMTIFF'],
                      
                      ['SI246120231031190932_',
                       'Vaildation print 31102023_SI246120231031190932_'])
        # obj = Model3D(openname, 
        #              ['C:/AAAWeichen/Mold (important!)/OT', 
        #               'C:/AAAWeichen/Mold (important!)/MPMTIFF'])
    
        obj.build(Range = [195, 249],
                  idxmask = [i for i in range(22)],
                  savename = model3Dname)
        obj.updateDepth(savename = model3Dname)
    
    # Slice = obj.CutSlice((354,1330), (1947,910), originalPath = model3Dname+ '_original.npy')
    Slice = obj.CutSlice((638,730), (1960,365), originalPath = model3Dname+ '_original.npy')
    fig = obj.Model_3D_dots(originalPath = model3Dname+ '_original.npy')
    
    # obj.ShowSlice(202)
    