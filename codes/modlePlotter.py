# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:03:16 2024

@author: Admin
"""

import numpy as np
from mayavi import mlab

Slices = np.load('../models/molde/Updated OT Depth.npy')
#slices = np.load('../models/molde/Original OT.npy')
sliceS = np.load('../models/molde/Initial Setting.npy')

slices = np.hstack((Slices, sliceS))


pixel_length_cm = np.float16(1e-2)  # cm
pixel_depth_cm = np.float16(30e-4)  # cm
mlab.figure(bgcolor=(0.9, 0.9, 0.9))

xx, yy, zz = np.meshgrid(np.arange(0, slices.shape[0]) * pixel_depth_cm,
                         np.arange(0, slices.shape[1]) * pixel_length_cm,
                         np.arange(0, slices.shape[2]) * pixel_length_cm,
                         indexing='ij')
src = mlab.pipeline.scalar_field(xx, yy, zz, slices)
vol = mlab.pipeline.iso_surface(src, color=(0.4, 0.4, 0.4))
vol.actor.property.opacity = 1
mlab.show()



# Resample the slices to represent the exact length and Create mesh grid
xx, yy, zz = np.meshgrid(np.arange(0, Slices.shape[0]) * pixel_depth_cm,
                         np.arange(0, Slices.shape[1]) * pixel_length_cm,
                         np.arange(0, Slices.shape[2]) * pixel_length_cm,
                         indexing='ij')
# Visualize using Mayavi
src = mlab.pipeline.scalar_field(xx, yy, zz, Slices)
vol = mlab.pipeline.iso_surface(src, color=(0.4, 0.4, 0.4))
vol.actor.property.opacity = 1
mlab.show()


xx, yy, zz = np.meshgrid(np.arange(0, sliceS.shape[0]) * pixel_depth_cm,
                         np.arange(0, sliceS.shape[1]) * pixel_length_cm,
                         np.arange(0, sliceS.shape[2]) * pixel_length_cm,
                         indexing='ij')
src = mlab.pipeline.scalar_field(xx, yy, zz, sliceS)
vol = mlab.pipeline.iso_surface(src, color=(0.4, 0.4, 0.4))
vol.actor.property.opacity = 1
mlab.show()