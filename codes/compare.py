# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:10:53 2024

@author: Admin
"""
from mayavi import mlab
from scipy.ndimage import median_filter
import numpy as np
from tqdm import tqdm
import cv2
import os

def idx2name(i,  OT_type = "Int"):
    '''
    To trasnlate loading index to 
    ------
    Parameters:
        centers: (list of np array) centers from last layer
        
    Returns:
        None
    '''
    OT_root = 'C:/AAAWeichen/Mold (important!)/OT/INT'
    OT_prefix = 'SI246120230324152157_'
    numb = (i + 1) * 3  
    ten = numb % 100
    hund = int((numb - ten)/100)
    MPM_name = str(hund).zfill(3)+ '_' + str(ten).zfill(2) + '0'
    OT_name = OT_prefix + str(i)+'_' + MPM_name +'_' \
        + OT_type + '_32F.tif'
    OT_name = os.path.join(OT_root, OT_name)
    return OT_name

def showvideo(idx = 0):
    opename = '../Files/'+str(idx)+'.npy'
    if os.path.exists(opename):
        Slices = np.load(opename)
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        indexs = list(range(len(Slices)))
        for i in tqdm(indexs):
            cv2.imshow('test', Slices[i])
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

def plot(idx = 0,  length = 1904, margin = 25, minaera = 100, show = False):
    # initialization
    opename = '../Files/'+str(idx)+'.npy'
    if False:#os.path.exists(opename):  # 
        Slices = np.load(opename)
    else:
        size=2000
        Mask = np.zeros((size,size),np.uint8)
        img = cv2.imread(idx2name(0), cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)
        # Find out all contours in threshold OT image
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Mark the initial part based on the first layer
        contour = sorted(contours, key = lambda x : x.size, reverse = True)[idx]
        # obtain the examin region from initial layer
        ctr = cv2.minAreaRect(contour)
        ctr = (ctr[0],(ctr[1][0] + margin, ctr[1][1] + margin), ctr[2])
        box = np.intp(cv2.boxPoints(ctr))
        xmin, ymin = box[:,1].min(), box[:, 0].min()
        xmax, ymax = box[:,1].max(), box[:, 0].max()
        # Get the Mask for the index region
        Mask = cv2.drawContours(Mask, [box], -1, (255), -1)
        # Initial the SLice region with size
        Slices = [np.zeros((xmax-xmin, ymax-ymin), np.uint8)]
        indexs = list(range(length))
        if show: # Check the region for caculating
            cv2.namedWindow(str(idx))
            cv2.imshow(str(idx),cv2.resize(Mask, (1000,1000)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for i in tqdm(indexs):
            filename = idx2name(i)
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(cv2.bitwise_and(img, Mask), 3, 255, cv2.THRESH_BINARY)
            img = img[xmin:xmax,ymin:ymax]
            contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = [_ for _ in contours if _.size < minaera]
            cv2.drawContours(img, contours, -1, (0), 1)
            if show:
                cv2.imshow(str(idx), img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if img.max() == 0:
                print('exit on' + filename)
                break
            return contours
            Slices.append(img)
        Slices.append(Slices[0])
        Slices = np.array(Slices)
        # Median Filter
        slices = Slices.copy()
        for z in range(Slices.shape[0]-2):
            slices[z+1, :, :] = median_filter(Slices[z+1, :, :], size=7)
        for z in range(Slices.shape[0]-2):
            Slices[z+1, :, :] = median_filter(slices[z+1, :, :], size=7)
        del slices
        np.save(opename, Slices)
        
    # Resample the slices to represent the exact length and Create mesh grid
    pixel_length_cm = np.float16(0.01)  # cm
    pixel_depth_cm = np.float16(0.003)  # cm
    xx, yy, zz = np.meshgrid(np.arange(0, Slices.shape[0]) * pixel_depth_cm,
                             np.arange(0, Slices.shape[1]) * pixel_length_cm,
                             np.arange(0, Slices.shape[2]) * pixel_length_cm,
                             indexing='ij')
    # Visualize using Mayavi
    mlab.figure(bgcolor=(0.9, 0.9, 0.9))
    src = mlab.pipeline.scalar_field(xx, yy, zz, Slices)
    return src

if __name__ == '__main__':
    src = plot(3, 1904)
    vol = mlab.pipeline.iso_surface(src, color=(0.4, 0.4, 0.4))
    vol.actor.property.opacity = 1   # Transparency
    mlab.show()
