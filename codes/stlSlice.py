# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:28:20 2024

@author: Admin
"""
import cv2
import meshio
from mayavi import mlab
import numpy as np

class Mode:
    def __init__(self, filename):
        self.filename = filename
        if filename[-3:] == 'stl':
            mesh = meshio.read(filename)
        # Get vertices and faces from the mesh
            self.vertices = mesh.points*10/3
            for i in range(3):
                self.vertices[:,i] -= self.vertices[:,i].min()
            self.vertices = np.round(self.vertices).astype(int)
            self.faces = mesh.cells_dict['triangle']
            self.shape = [self.vertices[:,i].max()+1 for i in range(3)]
            # Get numpy array
            self.array = np.zeros(self.shape, dtype = np.uint8)
            for i in range(self.vertices.shape[0]):
                self.array[tuple(self.vertices[i,:])] = 255
                ''' 
                    
            for i in range(self.array.shape[2]):
                #cv2.imshow("")
                contours, _ = cv2.findContours(self.array[:,:,i], cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                #cv2.drawContours(self.array[:,:,i],contours,-1, (255), cv2.FILLED)'''
    
    def show(self):
        for i in range(self.array.shape[2]):
            cv2.imshow("test", self.array[:,:,i])
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()  
    
#faces = mesh.cells_dict['triangle']

# vertices = mode.vertices
# Plot using Mayavi
''' 
fig = mlab.figure()
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], faces)
mlab.show()  
'''
'''
fig = mlab.figure(bgcolor=(0, 0, 0), size=(150, 150))
mlab.points3d(vertices[:,0], vertices[:,1], vertices[:,2],
                         vertices[:,2],  # Values used for Color
                         mode="point",
                         # 灰度图的伪彩映射
                         colormap='Spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )

''' 

if __name__ == '__main__':
    filename = 'C:\AAAWeichen\Mold (important!)\Mold original stl\Handle Latch V3 - 1.stl'
    mode = Mode(filename)
    vertices = mode.vertices
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(150, 150))
    mlab.points3d(vertices[:,0], vertices[:,1], vertices[:,2],
                             vertices[:,2],  # Values used for Color
                             mode="point",
                             # 灰度图的伪彩映射
                             colormap='Spectral',  # 'bone', 'copper', 'gnuplot'
                             # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                             figure=fig,
                             )
    mlab.show()
    
    #mode.show()
