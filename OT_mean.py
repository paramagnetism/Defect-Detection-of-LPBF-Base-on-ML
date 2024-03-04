# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:36:46 2024

@author: Wilson
"""
import numpy as np
import openpyxl
import cv2
from findrect import FindRect
from data2excel import XLS

class OT_mean(FindRect):
    def __init__(self, filename, rectnum):
        super(OT_mean, self).__init__(filename, rectnum)
        self.src = cv2.imread(filename)
        self._gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        self.__type__ = 'OT_mean'
    
    def cal_mean(self, gap = 0.05, show = False):
        step = (1+gap)/self.divpart
        self.mean = []
        count = 0 
        # For generating map
        self.ctrs = []
        #for show labeled order
        for x1, y1, x2, y2 in self._inner_rects:
            width = x2 - x1
            for i in range(self.divpart):
                xx1 = x1+i*step*width
                xx2 = x1+((i+1)*step-gap)*width
                corners = [(xx1, y1), (xx1, y2), (xx2, y2), (xx2, y1)]
                pts = [np.dot(self._M_, np.array([cnr[0], cnr[1], 1])) for cnr in corners]
                
                if show:
                    for j in range(4):
                        pt1 = (round(pts[j][0]), round(pts[j][1]))
                        pt2 = (round(pts[(j+1)%4][0]),round(pts[(j+1)%4][1]))
                        cv2.line(self.src, pt1, pt2, (0,255,0)) 
                    count+=1
                    cv2.putText(self.src, str(count), pt1, cv2.FONT_HERSHEY_COMPLEX, 
                                3/self.divpart, (0,0,255), 1)
                
                # use mask to choose the aera to be calculated
                mask = np.zeros(self._gray.shape).astype(np.uint8)
                pts = np.array(pts).astype(np.int32)
                cv2.fillConvexPoly(mask, pts, (255))
                self.mean.append(cv2.mean(self._gray, mask)[0])
                
                # The OT centers should be in 2500 scale
                self.ctrs.append(((pts[0]+pts[2])*0.625).astype(np.int32))
                
        if show:
            self._imshow(self.src)
    

if __name__ == '__main__':
    #findrect = OT_mean('./OT/OT/SI246120231031190932_239_007_200_Max_32F.tif', 27)
    findrect = OT_mean('E:/27 OT/Max/SI246120230518190639_0_000_030_Max_32F.tif', 27)
    #hist, hist2 = findrect.hist()
    # findrect.FullProcess()
    findrect.threshold(1, show = True)
    findrect.morphoperation(15, show = True)
    findrect.get_rotate()
    findrect.find_inner_rects()
    findrect.cal_mean(show = True) 
    findrect.matching_strip()
    # write in 
    #xls = XLS(loadname = "./OT/validation data V3.xlsx", loadpage = "Results")
    #xls = XLS(loadname = "./28Case/28_result.xlsx", loadpage = "Sheet1")
    #xls.save("./28Case/OT/OT_Max.xlsx", findrect.strip_color)
    #xls.save("./28Case/28RESULTT.xlsx",findrect.strip_color)
