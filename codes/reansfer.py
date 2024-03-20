# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:02:10 2024

@author: Admin
"""
import numpy as np
import cv2
import os

folder_path = 'C:/AAAWeichen/Mold (important!)/raw OT/'
for filename in os.listdir(folder_path):
    if filename.endswith('Int_32F.raw'):
        # Construct the full path to the raw file
        raw_file_path = os.path.join(folder_path, filename)
        
        with open(raw_file_path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
            image = raw_data.reshape(4000, 4000)
            cv2.imwrite('C:/AAAWeichen/Mold (important!)/OT/INT/'+filename[:-4] +'.tif', image)