# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:48:16 2024

@author: Wilson Li
"""
import os
import numpy as np
from MPM_mean import Pointlist, MPM_mean
from OT_mean import OT_mean
from data2excel import XLS
from tqdm import tqdm

def getCalib(rootdir, rectnum, axis):
    divpart = 5 if rectnum == 22 else 1
    means = [0 for i in range(rectnum*divpart)]
    ctrs = []
    img_count = 0
    for root, dirs, files in os.walk(rootdir):
        files.sort()
        filenames = [os.path.join(root, filename) for filename in files]
        if len(files) == 4:
            point_list = Pointlist(filenames[2], filenames[3], filenames[axis], params = 'anticlock')
            gray = point_list.imshow()
            findrectt = MPM_mean(gray, rectnum = rectnum)
            findrectt.FullProcess()
            findrectt.cal_mean(point_list, show = False) # 5 / 1
            mean = findrectt.mean
            means = [means[i] + mean[i] for i in range(rectnum*divpart)]
            img_count += 1
        else:
            for filename in tqdm(filenames):
                findrectt = OT_mean(filename, rectnum)
                findrectt.FullProcess(threshold = 1)
                findrectt.cal_mean(show = False)
                mean = findrectt.mean
                means = [means[i] + mean[i] for i in range(rectnum*divpart)]
                if img_count > 0:
                    for i, ctr in enumerate(findrectt.ctrs):
                        if np.sqrt(np.sum((ctr - ctrs[-1][i])**2)) > 100:
                            print('\n Error in number'+str(i))
                            findrectt.cal_mean(show = True)
                            
                ctrs.append(findrectt.ctrs)
                img_count += 1
    means = [Mean/img_count for Mean in means]
    return means
    # return findrectt.matching_strip()
    
if __name__ == '__main__':
    # for axis:  0 is off_axis, 1 is on axis
    #strip_color = getCalib('MPMfig', rectnum = 22, divpart = 5, axis = 1)
    #xls = XLS(loadname = "./OT/validation data V3.xlsx", loadpage = "Results")
    #xls.save("./MPM/optim_off.xlsx", strip_color)
    strip_color = getCalib('E:/27 OT/Max', rectnum = 27, axis = 0)
    #xls = XLS(loadname = "./28Case/28Result.xlsx", loadpage = "Sheet")
    #xls.save("./28Case/optim_off.xlsx", strip_color)
    

