# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:16:46 2024

@author: Admin
"""
import os
import cv2
import math
import pickle
import numpy as np
from tqdm import tqdm
from mayavi import mlab
import matplotlib.pyplot as plt

def idx2name(i,  OT_type = "Int"):
    '''
    To trasnlate loading index to 
    ------
    Parameters:
        centers: (list of np array) centers from last layer
        
    Returns:
        None
    '''
    OT_root = 'C:/AAAWeichen/Mold (important!)/OT/INT'
    OT_prefix = 'SI246120230324152157_'
    numb = (i + 1) * 3  
    ten = numb % 100
    hund = int((numb - ten)/100)
    MPM_name = str(hund).zfill(3)+ '_' + str(ten).zfill(2) + '0'
    OT_name = OT_prefix + str(i)+'_' + MPM_name +'_' \
        + OT_type + '_32F.tif'
    OT_name = os.path.join(OT_root, OT_name)
    return OT_name

IDX = 3
margin = 50
length = 1904
size = 2000
minaera = 100
thresh = 3

# of scanned image
right_adj = 1
up_adj = 0
# of origianl image
Right_adj = 2
Up_adj = 1
delay = 27
show = False
excluded = list(range(450, 491))+list(range(707, 715))+list(range(607, 619))+list(range(0,11))

Mask = np.zeros((size,size),np.uint8)
img = cv2.imread(idx2name(0), cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
# Find out all contours in threshold OT image
contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Mark the initial part based on the first layer
contour = sorted(contours, key = lambda x : x.size, reverse = True)[IDX]
# obtain the examin region from initial layer 
ctr = cv2.minAreaRect(contour)
ctr = (ctr[0],(ctr[1][0] + margin, ctr[1][1] + margin), ctr[2])
box = np.intp(cv2.boxPoints(ctr))
xmin, ymin = box[:,1].min(), box[:, 0].min()
xmax, ymax = box[:,1].max(), box[:, 0].max()
# Get the Mask for the index region
Mask = cv2.drawContours(Mask, [box], -1, (255), -1)
position = [0, 0, 0]; Position = [0, 0, 0]; 
for idx in range(20): 
    img = cv2.imread(idx2name(idx + 11 + delay), cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('../pngSlicer/outputs/png_filled/{0}.png'.format(idx + 11), cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(cv2.bitwise_and(img, Mask), thresh, 255, cv2.THRESH_BINARY)
    img = img[xmin:xmax,ymin:ymax]
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    small_contours = [_ for _ in contours if _.size < minaera]
    cv2.drawContours(img, small_contours, -1, (0), 1)
    contour = sorted(contours, key = lambda x : x.size, reverse = True)[0]
    Contour, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Contour = Contour[0]
    rect = cv2.minAreaRect(contour)
    Rect = cv2.minAreaRect(Contour)
    #print(rect)
    if rect[2] < 45:
        position[2] += (rect[2]+90)
    else: 
        position[2] += rect[2]
    position[0] += rect[0][0]
    position[1] += rect[0][1]
    Position[0] += Rect[0][0]
    Position[1] += Rect[0][1]
    if Rect[2] < 45:
        Position[2] += (Rect[2] + 90.0)
    else:
        Position[2] += Rect[2]
    
Position = [_/(idx+1) for _ in Position]  
position = [_/(idx+1) for _ in position]    

rows, cols = img.shape[:2]
ctr = (position[0], position[1])
angle = position[2] - Position[2]
          
rotation_matrix = cv2.getRotationMatrix2D(ctr, angle, 1.0)
# Calculate the new bounding dimensions to ensure the whole image fits
abs_cos = abs(rotation_matrix[0, 0])
abs_sin = abs(rotation_matrix[0, 1])

# Compute the new bounding dimensions of the image
w_ = rows * abs_sin + cols * abs_cos
h_ = rows * abs_cos + cols * abs_sin
bound_w = round(w_)
bound_h = round(h_)

# Adjust the translation matrix to consider the new dimensions      
rotation_matrix[0, 2] += w_ // 2 - ctr[0]
rotation_matrix[1, 2] += h_ // 2 - ctr[1]
# Caculate the transformmed coordinate
bound = [round(Position[0]-w_/2),round(Position[0]-w_/2)+bound_w, 
         round(Position[1]-h_/2),round(Position[1]-h_/2)+bound_h]

buffer = np.zeros((bound[3]-bound[2], bound[1]-bound[0]))
with open('../models/modelsImageD/OT_Int (0-255).pkl',"rb") as f:
    model = pickle.load(f)['model']
with open('../models/Dskin.pkl',"rb") as f:
    dskin = pickle.load(f)
constDepth = dskin.predict(np.array([[370, 1300],]))    

summ = 0
transformed = []; before = []; fly = []; idxs = []
begin = 0
end = 736 - delay
downskin = 40

Slices = [np.zeros((bound_h, bound_w), np.uint8)]
slices = [np.zeros((bound_h, bound_w), np.uint8)]
sliceS = [np.zeros((bound_h, bound_w), np.uint8)]
maximprov = 0
circumstances = []
b4Biases = []
for idx in tqdm(reversed(range(736 - delay))): 
    img = cv2.imread(idx2name(idx+delay), cv2.IMREAD_GRAYSCALE)
    org = cv2.imread('../pngSlicer/outputs/png_original/{0}.png'.format(idx + delay + 1), cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('../pngSlicer/outputs/png_filled/{0}.png'.format(idx), cv2.IMREAD_GRAYSCALE)
    scanned_image = src[bound[2]+up_adj:bound[3]+up_adj,
                    bound[0]+right_adj:bound[1]+right_adj]
    original_image = org[bound[2]+Up_adj:bound[3]+Up_adj,
                        bound[0]+Right_adj:bound[1]+Right_adj]
    # Apply the affine transformation to the image
    img = cv2.bitwise_and(img, Mask)[xmin:xmax,ymin:ymax]                                              
    img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
    _, OTFly = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        
    img[original_image == 0] = 0 
    
    # Caculate the depth for contours
    _, transformed_image = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    b4 = transformed_image.copy()
    contours, _ = cv2.findContours(transformed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(original_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    Depth = np.zeros(transformed_image.shape).astype(np.uint8)
    circumstance = 0
    for contour in contours:
        msk = np.zeros(transformed_image.shape).astype(np.uint8)
        cv2.drawContours(msk, [contour], -1, (255), 1)
        OT_strip = cv2.mean(img, msk)[0]
        Depth[msk>0] = model.predict(np.array([[OT_strip, 370, 1300],])) + constDepth
        
    for contour in contours2:
        circumstance += cv2.arcLength(curve=contour, closed=True)
        
    buffer = np.maximum(buffer-30, Depth) 
    transformed_image[buffer > 0] = 255 
    
    SNR = 10* math.log((scanned_image>0).sum()/(transformed_image != scanned_image).sum(),10)
    flyed = 10* math.log((scanned_image>0).sum()/(OTFly != scanned_image).sum(),10)
    aveBias = ((transformed_image!=scanned_image)>0).sum()/circumstance/8
    b4Bias = ((b4!=scanned_image)>0).sum()/circumstance/8
    # print('Image {}, SNR {}, circumstance {}'.format(idx, SNR, aveBias))
    # print(SNR)
    if idx not in excluded: 
        idxs.append(idx)
        transformed.append(SNR)#circumstance/16)
        before.append(10 * math.log((scanned_image>0).sum()/(b4 != scanned_image).sum(), 10))#circumstance/16)
        fly.append(flyed)#circumstance/16)      
        circumstances.append(aveBias)      
        b4Biases.append(b4Bias)
        if idx < 40 and flyed < 0.8 *SNR and SNR - flyed > maximprov:
            maximprov = flyed - SNR 
            compare = np.zeros(scanned_image.shape+(3,))
            cv2.imwrite('Images/original.png', original_image)
            cv2.imwrite('Images/transformed.png', transformed_image)
            cv2.imwrite('Images/scanned.png', scanned_image)
            compare[:,:,0] = transformed_image  # B
            compare[:,:,2] = scanned_image   # G
            #compare[:,:,2] = original_image  # R 
            #compare[:,:,0][scanned_image/2+original_image/2>200]=255 
            #compare[:,:,1][transformed_image/2+original_image/2>200]=255
            #compare[:,:,2][transformed_image/2+scanned_image/2>200]=255 
            compare[:,:,1] = transformed_image #[transformed_image/2 + scanned_image/2>200]=255 
            cv2.imwrite('Images/scannedVStransformed.png', compare)
            
            compare2 = np.zeros(scanned_image.shape+(3,))
            compare2[:,:,2] = scanned_image  # B
            compare2[:,:,1] = OTFly
            compare2[:,:,0][OTFly/2 + scanned_image/2 > 200]=255
            cv2.imwrite('Images/scannedVSoriginal.png', compare2)
            
        if  idx< 60  and show: #aveBias > 0.10
            # transformed image: OT image result
            # scanned: XCT slices result
            # Fly: OT without cut
            # original: XCT slices initial setting
            compare = np.zeros(scanned_image.shape+(3,))
            compare[:,:,0] = scanned_image  # B
            compare[:,:,1] = transformed_image   # G
            compare[:,:,2][scanned_image/2 + transformed_image/2>200]=255 
            cv2.imshow('{}'.format(idx),compare)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                end = idx
                break
    # print('Fly:{0}; Before:{1}; transformed:{2}'.format(fly, before, transformed))
    # print(fly)
    # Demo part
        
    #cv2.waitKey(0)
    Slices.append(transformed_image)
    slices.append(OTFly)
    sliceS.append(scanned_image)
    
slices.append(slices[0])
Slices.append(Slices[0])
sliceS.append(sliceS[0])
Slices = np.array(Slices)
slices = np.array(slices)
sliceS = np.array(sliceS)
np.save('../models/molde/Updated OT Depth', Slices)
np.save('../models/molde/Original OT', slices)
np.save('../models/molde/Initial Setting', sliceS)
cv2.destroyAllWindows()

x = np.array(idxs)
fig, axe = plt.subplots()
axe.plot(x, np.array(fly), x, np.array(transformed))
plt.xlabel("Layers indexs @ delay:{}".format(delay))
plt.ylabel("SNR(dB)")
plt.legend(['Raw In-situ Image model','Reconstructed model'], loc="upper right")
plt.show()

Improve = (np.array(fly)-np.array(transformed)).mean()
Mean1 = np.array(fly).mean()
Mean2 = np.array(transformed).mean()

xx = x[-downskin:]
fig, axe = plt.subplots()
axe.plot(xx, np.array(fly[-downskin:]), xx, np.array(transformed[-downskin:]), xx, np.array(before[-downskin:]))
plt.xlabel("Layers indexs on Downskin Region")
plt.ylabel("SNR(dB)")
plt.legend(['Raw In-situ Image model','Reconstructed model', 'Denoised Only model'], loc="upper left")
plt.show()

Improve1 = (np.array(fly[-downskin:])-np.array(before[-downskin:])).mean()
Improve2 = (np.array(fly[-downskin:])-np.array(transformed[-downskin:])).mean()
mean1 = np.array(fly[-downskin:]).mean()
mean2 = np.array(transformed[-downskin:]).mean()
mean3 = np.array(before[-downskin:]).mean()

circumstances = np.array(circumstances)
b4Biases = np.array(b4Biases)
ave = circumstances.mean()
dskinave = circumstances[-50:].mean()
#print(delay)
#print(dskinave)
#print(circumstances[-50:].max())
Max = max(circumstances)
fig, axe = plt.subplots()
axe.plot(x, circumstances)
# axe.plot(x, b4Biases)
axe.plot(x, np.ones(circumstances.shape)*ave, ls = '--')
axe.plot(x, np.ones(circumstances.shape)*0.1, ls = '--')
plt.ylabel("Average error(mm)")

plt.xlabel("Layer no")
plt.title("Distribution of error across layers")
plt.show()
