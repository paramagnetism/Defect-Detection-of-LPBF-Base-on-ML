# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:32:08 2024

@author: Wilson

This file is used to find contours and caculate depth on OT and MPM image.

"""
   
import numpy as np
import cv2
from data2excel import csv_read

class FindContour:
    """
    The most general method for caculating OT and MPM image.
    
    
    """
    def __init__(self, filename, rectnum):
        self.rectnum = rectnum
        self.__type__ = 'val_mean'
        self._gray = np.zeros((2,2))
        self.erosion = np.zeros((2,2)) 
        self.strip_color = []
        if type(filename) == str:
            self.src = cv2.imread(filename)
            # resize to shape of MPM
            self.src = cv2.resize(self.src,(2500,2500))
            self._gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        elif type(filename) == np.ndarray:
            self.Set(filename)
        
    def Set(self, img):
        if len(img.shape) == 2:
            self.src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self._gray = img
        else:
            self.src = img
            self._gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    # show the image in 1/2 size for the screen
    def _imshow(self, img):
        img = cv2.resize(img,(img.shape[1]>>1, img.shape[0]>>1))
        cv2.imshow("show",img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    # Plot hist to observe for a good threshold
    def hist(self): 
        hist = cv2.calcHist([self._gray], channels = [0], mask = None, 
                            histSize = [256], ranges = [0, 256])
        aft_hist = cv2.equalizeHist(self._gray)
        hist2 = cv2.calcHist([aft_hist], channels = [0], mask = None, 
                             histSize = [256], ranges = [0, 256])
        return hist, hist2
    
    # Threshold operation, usually choose thresh = 2
    def threshold(self, threshold = 2, show = False):
        ret, self.erosion = cv2.threshold(self._gray, threshold, 255, cv2.THRESH_BINARY)
        if show:
            self._imshow(self.erosion)
    
    # close operation removing noise
    # the kernel should be bigger with larger threshold
    def morphoperation(self, close_kernel_size = 3, show = False): 
        # kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        self.erosion = cv2.medianBlur(self.erosion, close_kernel_size)
        # self.erosion = cv2.morphologyEx(self.erosion, cv2.MORPH_CLOSE, kernel = kernel)
        if show:
            self._imshow(self.erosion)
        
    # Sort the founded rectangles and find biggest sizes
    def get_contours(self, show = False):
        self.contours, _ = cv2.findContours(self.erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [_ for _ in self.contours if _.size > 200]
        self.contours = sorted(self.contours, key = lambda x : x.size, reverse = True)[:self.rectnum]
        MM = [cv2.moments(contour) for contour in self.contours]
        self.ctrs = [np.array([int(M["m10"] / M["m00"]), 
                               int(M["m01"] / M["m00"])]) for M in MM]
        if show:
            showcase = self.src.copy()
            cv2.drawContours(showcase, self.contours, -1, (0,255,0), 1)
            for i, contour in enumerate(self.contours):
                cv2.putText(showcase, str(i),self.ctrs[i], cv2.FONT_HERSHEY_COMPLEX, 
                            3, (0,0,255), 1)
            self._imshow(showcase)
     
    def cal_mean(self):
        self.mean = []
        for contour in self.contours:
            mask = np.zeros_like(self._gray)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            self.mean.append(cv2.mean(self._gray, mask=mask)[0])
         
            
    def FullProcess(self, threshold = 2, close = 3, show = False):
        self.threshold(threshold,show = show)
        self.morphoperation(close, show = show)
        self.get_contours(show = show)
        self.cal_mean()


class FindRect(FindContour):
    def __init__(self, filename, rectnum):
        super().__init__(filename, rectnum)
        self.divpart = 5 if rectnum == 22 else 1
        self.sorting_slope = 10 if rectnum == 22 else 0.55
        self._max_black = 10
           
            
    def get_rotate(self, show = False):
        super().get_contours(show = show)
        # Find mean rotation angle for all fitted rotate rect
        Mean_angle = np.array([cv2.minAreaRect(cnt)[2]-90 for cnt in self.contours]).mean()
        # Get Shift and rotate transform matrix M, and inverse M_
        # Scale little a bit to accomend to shifted image
        self._M = cv2.getRotationMatrix2D([self.src.shape[1]/2, 
                                           self.src.shape[0]/2], Mean_angle, 0.8)
        self._M_ = cv2.getRotationMatrix2D([self.src.shape[1]/2, 
                                            self.src.shape[0]/2], - Mean_angle, 1.25)
        # Do Affine transform, .img is the transformed vertical plot
        self.__rotated_img = cv2.warpAffine(self.erosion, self._M, 
                                            (self.src.shape[0], self.src.shape[1]))


    def find_inner_rects(self, show = False):
        #find contours in rects plotted in vertical angle
        contours2, _ = cv2.findContours(self.__rotated_img, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours2 = [_ for _ in contours2 if _.size > 8]
        contours2 = sorted(contours2, key = lambda x : cv2.contourArea(x), 
                           reverse = True)[:self.rectnum]
        # need to arrange for this initial order
        # sorting slope = 10 for 110 OT img  , >1 for row sort, < 1 for line sort
        # sorting slope = 0.55 for 27 OT img , slope = 1/math.atan(-findrect.mean_angle)-0.1
        self._inner_rects = [maximum_internal_rectangle(self.__rotated_img, cnt,
                                                        self._max_black) for cnt in contours2]
        self._inner_rects = sorted(self._inner_rects, key = lambda x: x[0]*self.sorting_slope-x[1])

        if show:
            count = 1
            showcase = cv2.cvtColor(self.__rotated_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(showcase, contours2, -1, (0,255,0), 1)
            for x1, y1, x2, y2 in self._inner_rects:
                cnr1, cnr2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(showcase, cnr1, cnr2, (255, 0, 0), 2)
                cv2.putText(showcase, str(count), cnr2, 
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 1)
                count += 1
            self._imshow(showcase)
        # return self._inner_rects
    
    # Matching name 
    def matching_strip(self):
        # determine the name is OT or MPM
        valname = self.__type__ if self.__type__[0] == 'O' else self.__type__[:-6]
        count = 0
        # 22 Rect
        if self.rectnum == 22:  
            for line in range(5): 
                rows_num = 6 if line == 0 else 4
                for row in range(rows_num):
                    for strip in range(self.divpart):
                        rst = {'Degree':line*10+45, 
                               'Part No.':row+1, 
                               'Strip No. on part':strip+1,
                               valname : self.mean[count]}
                        self.strip_color.append(rst)
                        count += 1
        # 27 rect
        elif self.rectnum == 27:
            for name in ['N_', '45Deg_', '30Deg_']:
                for i in range(9):
                    rst = {'No. Part': name + str(i+1),
                           valname : self.mean[count]}
                    self.strip_color.append(rst)
                    count += 1  
            
        # return self.strip_color    
    def FullProcess(self, threshold = 3, close = 15):
        self.threshold(threshold)
        self.morphoperation(close)
        self.get_rotate()
        self.find_inner_rects()


class Pointlist:
    def __init__(self, filename_x, filename_y, filename_val, params = ''):
        dot_val = csv_read(filename_val)
        dot_x = csv_read(filename_x)
        dot_y = csv_read(filename_y)
        self.list = []
        self.params = params
        for i, val in enumerate(dot_val):
            self.list.append(self.Point(dot_x[i], dot_y[i], val, self.params))
            
    class Point:
        def __init__(self, x, y, val, params):
            self.val = val
            if '-Y' in params:
                self.y = 2500 - y * 10
            else: self.y = y * 10
            if '-X' in params:
                self.x = 2500 - x * 10
            else: self.x = x * 10
            if 'clockwise' in params:
                self.x, self.y = 2500 - self.y, self.x
            if 'anticlock' in params:
                self.x, self.y = self.y, 2500 - self.x
    
    def imshow(self, show = False):
        count = np.zeros([2500,2500])
        img = np.zeros([2500,2500])
        for dot in self.list:
            x = round(dot.x)
            y = round(dot.y)
            count[y,x] += 1
            img[y,x] += dot.val
        image = np.divide(img, count, out=np.zeros_like(img), where = count!=0)
        self.image = np.rint(255 * image / image.max()).astype(np.uint8)
        if show:
            size = (int(0.5*self.image.shape[0]),int(0.5*self.image.shape[1]))
            img = cv2.resize(self.image,size,fx=1,fy=1)
            cv2.imshow("show",img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return self.image #np.rint(self.__rotated_img).astype(np.uint8)
            


def maximum_internal_rectangle(img, cnt, max_black = 50): 
    rect = []
    contour = cnt.reshape(len(cnt),2)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    
    # Calculating all possible aera
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))
    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)

        while not best_rect_found and index_rect < nb_rect:
 
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
            valid_rect = True
 
            x = min(x1, x2)
            # Contours object only contain the outest layer of the shape
            # If the vertex of the rextangle doesn't lie on the contour 
            # the shape can be rejected if the shape is non convex
            # adding black pixel count to allow non convex to some extend
            black_count = 0 
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    black_count += 1
                    if black_count > max_black:
                        valid_rect = False
                x += 1
 
            y = min(y1, y2)
            black_count = 0 
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    black_count += 1
                    if black_count > max_black:
                        valid_rect = False
                y += 1
 
            if valid_rect:
                best_rect_found = True
 
            index_rect += 1
 
        if best_rect_found:
            x1, x2 = min(x1,x2), max(x1, x2)
            y1, y2 = min(y1,y2), max(y1, y2)
            return x1, y1, x2, y2

        else:
            print("No rectangle fitting into the area")
            return 0, 0, 0, 0
    else:
        print("No rectangle found")
        return 0, 0, 0, 0
    
if __name__ == '__main__':
    pass
