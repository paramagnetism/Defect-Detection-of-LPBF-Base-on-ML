# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:28:20 2024

@author: Admin
"""
import cv2
import meshio
from mayavi import mlab
import numpy as np

# from stl import mesh
filename = 'C:\AAAWeichen\Mold (important!)\AM Mould Insert (GOM scan) (2)\Mould insert_F_standard.stl'
# model = np.array(mesh.Mesh.from_file(filename).vectors)
mesh = meshio.read(filename)

# Get vertices and faces from the mesh
vertices = mesh.points
faces = mesh.cells_dict['triangle']


# Plot using Mayavi
fig = mlab.figure()
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
mlab.show()

