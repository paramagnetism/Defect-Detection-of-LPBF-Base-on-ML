# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:18:27 2024

@author: Wilson Li
"""
import numpy as np
import cv2
from findrect import FindRect, Pointlist
from data2excel import XLS

class MPM_mean(FindRect):
    # use generated image
    def __init__(self, filename, rectnum):
        super(MPM_mean, self).__init__(filename, rectnum)
        if type(filename) == str:     
            self.src = cv2.imread(filename)
            self._gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
            self.__type__ = 'MPM_mean_graph'
        else:  
            self.src = cv2.cvtColor(filename, cv2.COLOR_GRAY2BGR)
            self._gray = filename
            self.__type__ = 'MPM_mean_excel'
        
    def cal_mean(self, pointlist, gap = 0.05, show = False):
        step = (1+gap)/self.divpart
        self.strips = [[] for i in range(self.divpart*len(self._inner_rects))]
        # see if the labels are right
        if show or self.__type__ == 'MPM_mean_graph':
            count = 0
            self.mean = []
            self.ctrs = []
        # to show labeled order
            for x1, y1, x2, y2 in self._inner_rects:
                width = x2 - x1
                for i in range(self.divpart):
                    xx1 = x1+i*step*width
                    xx2 = x1+((i+1)*step-gap)*width
                # strip coordinate
                    corners = [(xx1, y1), (xx1, y2), (xx2, y2), (xx2, y1)]
                    pts = [np.dot(self._M_, np.array([cnr[0], cnr[1], 1])) for cnr in corners]
                    if show:
                        for j in range(4):
                            pt1 = (round(pts[j][0]), round(pts[j][1]))
                            pt2 = (round(pts[(j+1)%4][0]),round(pts[(j+1)%4][1]))
                            cv2.line(self.src, pt1, pt2, (0,255,0)) 
                        count+=1
                        cv2.putText(self.src, str(count), pt2, cv2.FONT_HERSHEY_COMPLEX, 
                                    3/self.divpart, (0,0,255), 1)
                    if self.__type__ == 'MPM_mean_graph':
                        mask = np.zeros(self._gray.shape).astype(np.uint8)
                        pts = np.array(pts).astype(np.int32)
                        cv2.fillConvexPoly(mask, pts, (255))
                        # exclude pixel without signal
                        mask[self._gray < 2] = 0
                        self.mean.append(cv2.mean(self._gray, mask)[0])
                        self.ctrs.append(((pts[0]+pts[2])/2).astype(np.int32))
                        
        if show: self._imshow(self.src) 
        if self.__type__ == 'MPM_mean_excel':
            for point in pointlist.list:
                # coordinate in vertical image
                [x, y] = np.dot(self.M, np.array([point.x, point.y, 1]))
                for j, (x1, y1, x2, y2) in enumerate(self._inner_rects):
                    if y1 <= y and y2 >= y:
                        width = x2 - x1
                        for i in range(self.divpart):
                            xx1 = x1+i*step*width
                            xx2 = x1+((i+1)*step-gap)*width
                            if xx1 <= x and xx2 >= x:
                                self.strips[i+self.divpart*j].append(point.val)
                                break
                        else:
                            continue
                        break
                else:
                    continue  
            self.mean = [sum(strip)/len(strip) for strip in self.strips]


if __name__ == '__main__':
    typ = "on"
    
    # point_list = Pointlist('./MPM/007_200_00-x.csv', './MPM/007_200_00-y.csv' ,'./MPM/007_200_00-'+typ+'-axis.csv', params = '-Y')
    # point_list = Pointlist('./28Case/MPM/mpm_x.csv', './28Case/MPM/mpm_y.csv' ,'./28Case/MPM/mpm_'+typ+'.csv', params = 'anticlock')
    # gray = point_list.imshow(show = True)
    # findrect = MPM_mean(gray, rectnum = 27)
    findrect = MPM_mean('MPM/Vaildation print 31102023_SI246120231031190932_007_200_00.mpm_ma_'+typ+'axis.tif', rectnum = 22)
    # hist, hist2 = findrect.hist()
    findrect.FullProcess()
    
    #findrect.cal_mean(point_list, show = True)
    findrect.cal_mean([], show = True) # 5 / 1
    
    findrect.matching_strip()
    
    xls = XLS(loadname = "./OT/validation data V3.xlsx", loadpage = "Results")
    #xls = XLS(loadname = "./28Case/28_result.xlsx", loadpage = "Sheet1")
    #xls.save("./28Case/MPM/result-"+typ+".xlsx", findrect.strip_color)
    xls.save("./MPM/Imgresult-"+typ+".xlsx", findrect.strip_color)