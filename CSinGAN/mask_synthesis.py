#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import PIL.Image as Image
import os
import random
import numpy as np
import scipy.io as io
import cv2
import os.path as osp
import math
import matplotlib.pyplot as plt
import scipy.io as scio
get_ipython().run_line_magic('matplotlib', 'inline')
def draw_masks(indexs,path,i):
    height = 300
    width = 300
    if not os.path.exists(path+str(i)):
        os.mkdir(path+str(i))
    save_path = path+str(i)
    for j in range(len(indexs)):
        zero =  np.zeros((height, width), dtype=np.uint8)
        contours = np.array(indexs[j])
        contours[:, [0, 1]] = contours[:, [1, 0]]
        mask = cv2.drawContours(zero, [contours], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)
        mask = Image.fromarray(mask[22:-22,22:-22], mode='L')
        if np.count_nonzero(np.array(mask)) < 20:
            continue
        mask.save(osp.join(save_path,str(j)+".png"))

def draw_whole_mask(indexs):
    height = 300
    width = 300
    whole = np.zeros((height, width), dtype=np.uint8)
    labelled = np.zeros((height, width), dtype=np.uint8)
    
    for j in range(len(indexs)):
        contours = np.array(indexs[j])
        contours[:, [0, 1]] = contours[:, [1, 0]]
        whole = cv2.drawContours(whole, [contours], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)
        labelled = cv2.drawContours(labelled, [contours], contourIdx=-1, color=j+1, thickness=cv2.FILLED)
    return whole,labelled
def distance(point_1, point_2):
    d=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    return d
def judge(countour,Range):
    for point in contour:
        x = point[0][0]
        y = point[0][1]
        if x in range(256-Range,Range) and y in range(256-Range,Range):
            continue
        else:
            return False
    return True
def move(cx,cy,whole,contour):
    point_range = []
    random_dis = [2,3,4,5]
    d_max = 0
    for point in contour:
        #print(point)
        d = distance(point[0],[cx,cy])
        if d > d_max:
            d_max = d
    #print('The max radius is: ',d_max)
    extra = random.choice(random_dis)
    #kernel = np.ones((int(d_max+extra),int(d_max+extra)),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(d_max+extra),int(d_max+extra)))
    dilation = cv2.dilate(whole,kernel,iterations = 1)
    #plt.figure(figsize=(2.5,2.5))
    #plt.imshow(dilation*255,cmap='gray')
    for i in range(300):
        for j in range(300):
            if dilation[i,j] == 0:
                point_range.append([i,j])
    #print(len(point_range))
    [x,y] = random.choice(point_range)
    #print([x,y])
    dx = x-cx
    dy = y-cy
    new_contour = []
    for point in contour:
        #print(point)
        n_x = point[0][0] + dx
        n_y = point[0][1] + dy
        new_contour.append([n_x,n_y])
    return new_contour

save_path = "/data1/partitionA/TMI/code/patch_selection/monuseg/cps9/draw_masks/labels/"
whole_path = "/data1/partitionA/TMI/code/patch_selection/monuseg/cps9/draw_masks/whole/"
select = ['62376', '46511', '70724', '5493', '74638', '16286', '57026', '34826', '14721', '24918', '66915', '64859']
select = sorted(select, key=lambda x: int(x))
random.seed( 21 )
dense = [[0.2,0.4],[0.4,0.6]]
dense_attri = [0,0,0,0,0,0,0,1,1,1,0,1]
for s in range(len(select)):
    path = "/data1/partitionA/TMI/code/patch_selection/monuseg/cps9/cps/"+str(select[s])+'_'
    h = dense_attri[s]


    all_nucleis = []
    for f in range(6):
        img_path = path + str(f) + "/masks/"
        files = os.listdir(img_path)
        for file in files:
        #print(img_path + file)
            img = cv2.imread(img_path + file,0)
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                if len(contours[0]) > 5:
                    all_nucleis.append(contours[0])

    height = 300
    width = 300
#ratio = 0.35
    Range = 250
 #21/100/500/1000
    n = s
    for i in range(50):
        whole = np.zeros((height, width), dtype=np.uint8)
    #files = os.listdir(img_path)
        length = len(all_nucleis)
        count = length//7
        print('real numbers:',count)
    
        random_count =  random.randint(int(dense[h][0]*count),int(dense[h][1]*count))
        countour_list = []
        center_list = []

        while True:
            number = random.randint(0,length-1)
            contour = all_nucleis[number]
            if len(contour) > 5:
                if judge(contour,Range):
                    M = cv2.moments(contour)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    new_contour = move(cx,cy,whole,contour)
                    countour_list.append(new_contour)
                    whole,labelled = draw_whole_mask(countour_list)

                #plt.figure(figsize=(2.5,2.5))
                #plt.imshow(whole*255,cmap='gray')
                    if len(countour_list) == (count + random_count):
                        #print(len(countour_list))
                        break
        
        whole = Image.fromarray(whole[22:-22,22:-22], mode='L')
    #labelled = Image.fromarray(labelled[22:-22,22:-22], mode='L')
        whole.save(osp.join(whole_path,str(i+n*50)+".png"))
    #dataNew =whole_path + str(i+n*50) + '.mat'
    #scio.savemat(dataNew, {'binary_mask':labelled[22:-22,22:-22]})
        draw_masks(countour_list,save_path,i+n*50)


# In[ ]:


select = ['62376', '46511', '70724', '5493', '74638', '16286', '57026', '34826', '14721', '24918', '66915', '64859']
select = sorted(select, key=lambda x: int(x))


# In[ ]:


select


# In[ ]:




