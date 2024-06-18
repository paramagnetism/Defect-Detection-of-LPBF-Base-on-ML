# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:48:16 2024

@author: Wilson Li
"""
import os
from MPM_mean import Pointlist, MPM_mean
from OT_mean import OT_mean
from data2excel import XLS
from tqdm import tqdm

def getCalib(rootdir, rectnum, axis, mode = 'MPM'):
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
                findrectt = MPM_mean(filename, rectnum) \
                    if mode == 'MPM' else OT_mean(filename, rectnum)

                findrectt.FullProcess()
                    
                findrectt.cal_mean([],show = False) \
                    if mode == 'MPM' else findrectt.cal_mean(show = False)
                findrectt.matching_strip()
                ''' 
                mean = findrectt.mean
                means = [means[i] + mean[i] for i in range(rectnum*divpart)]
                
                if img_count > 0:
                    for i, ctr in enumerate(findrectt.ctrs):
                        if np.sqrt(np.sum((ctr - ctrs[-1][i])**2)) > 100:
                            print('\n Error in number'+str(i))
                            findrectt.cal_mean([],show = True) \
                                if mode == 'MPM' else findrectt.cal_mean(show = True)
                ctrs.append(findrectt.ctrs)
                '''
                ctrs.append(findrectt.strip_color)
                img_count += 1
    means = ctrs[0]
    for mean in means:
        for Ctr in ctrs[1:]:
            for ctr in Ctr:
                if 'Degree' in ctr.keys():
                    if ctr['Degree'] == mean['Degree'] and \
                        ctr['Part No.'] == mean['Part No.'] and\
                            ctr['Strip No. on part'] == mean['Strip No. on part']:
                                mean['MPM_mean'] += ctr['MPM_mean']
                elif ctr['No. Part'] == mean['No. Part']:
                    mean['MPM_mean'] += ctr['MPM_mean']
        mean['MPM_mean'] /= len(ctrs)
    return means
    # return findrectt.matching_strip()
    
if __name__ == '__main__':
    # for axis:  0 is off_axis, 1 is on axis
    #strip_color = getCalib('E:/Calib/OFF/', rectnum = 22, axis = 1)
    #xls = XLS(loadname = "./OT/validation data V3.xlsx", loadpage = "Results")
    #xls.save("./MPM/optim_off.xlsx", strip_color)
    
    strip_color = getCalib('E:/27 MPM figures/Calib/OFF/', rectnum = 27, axis = 0)
    xls = XLS(loadname = "./28Case/28_result.xlsx", loadpage = "Sheet1")
    xls.save("./28Case/MPM/optim_off.xlsx", strip_color)
    
    strip_color = getCalib('E:/27 MPM figures/Calib/ON/', rectnum = 27, axis = 1)
    xls = XLS(loadname = "./28Case/28_result.xlsx", loadpage = "Sheet1")
    xls.save("./28Case/MPM/optim_on.xlsx", strip_color)
    
    #strip_color = getCalib('E:/27 OT/Max', rectnum = 27, axis = 0)
    #xls = XLS(loadname = "./28Case/28Result.xlsx", loadpage = "Sheet")
    #xls.save("./28Case/optim_off.xlsx", strip_color)
    

