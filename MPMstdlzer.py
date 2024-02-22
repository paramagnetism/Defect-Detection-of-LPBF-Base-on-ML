# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:48:16 2024

@author: Wilson Li
"""
import os
from MPM_mean import Pointlist, MPM_mean
from data2excel import XLS

def getCalib(rootdir, rectnum, axis):
    divpart = 5 if rectnum == 22 else 1
    means = [0 for i in range(rectnum*divpart)]
    img_count = 0
    for root, dirs, files in os.walk(rootdir):
        if len(files) == 4:
            files.sort()
            filenames = [os.path.join(root, filename) for filename in files]
            point_list = Pointlist(filenames[2], filenames[3], filenames[axis], params = 'anticlock')
            gray = point_list.imshow()
            findrectt = MPM_mean(gray, rectnum = rectnum)
            #hist, hist2 = findrectt.hist()
            findrectt.FullProcess()
            print('Processing '+root)
            # last check for every image
            findrectt.cal_mean(point_list, show = True) # 5 / 1
            mean = findrectt.mean
            means = [means[i] + mean[i] for i in range(rectnum*divpart)]
            img_count += 1
    means = [Mean/img_count for Mean in means]
    return means
    # return findrectt.matching_strip()
    
if __name__ == '__main__':
    # for axis:  0 is off_axis, 1 is on axis
    #strip_color = getCalib('MPMfig', rectnum = 22, divpart = 5, axis = 1)
    #xls = XLS(loadname = "./OT/validation data V3.xlsx", loadpage = "Results")
    #xls.save("./MPM/optim_off.xlsx", strip_color)
    strip_color = getCalib('28Case/MPMfig', rectnum = 27, axis = 0)
    xls = XLS(loadname = "./28Case/28Result.xlsx", loadpage = "Sheet")
    xls.save("./28Case/optim_off.xlsx", strip_color)
    

