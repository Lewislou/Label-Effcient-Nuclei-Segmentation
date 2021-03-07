#!/usr/bin/env python
# coding: utf-8

# In[3]:


##TSNE

import os
import numpy as np
import shutil
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
img_to_tensor = transforms.ToTensor()

def make_model():
    resnet=models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]     
    #GAP = nn.AdaptiveAvgPool2d((1,1))
    model = nn.Sequential(*modules)
    model=model.eval()
    model.cuda()
    return model
    

def extract_feature(model,tensor):
    model.eval()
    tensor=tensor.cuda()   
    result=model(Variable(tensor))
    result_npy=result.data.cpu().numpy()
    
    return result_npy

def select_masks(mask,path):
    max_value = mask.max()
    indexs = []
    for i in range(1,max_value+1):
        x = np.where(mask == i)
        index = np.asarray(x).transpose(1,0).tolist()
        if len(index) > 0:
            indexs.append(index) 
    height = 128
    width = 128
    for j in range(len(indexs)):
        zero =  np.zeros((height, width), dtype=np.uint8)
        contours = np.array(indexs[j])
        contours[:, [0, 1]] = contours[:, [1, 0]]
        mask = cv2.drawContours(zero, [contours], contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)
        mask = Image.fromarray(mask, mode='L')
        mask.save(osp.join(path,str(j)+".png"))

def crop(image,mask,path_crop,c,i):

    path = path_crop+str(c)+'\\' ##new path to save cropped data
    if not os.path.exists(path):
        os.mkdir(path)
    for x in range(0,256,128):
        for y in range(0,256,128):
            image_crop = np.array(image)[x:x+128,y:y+128,:]
            mask_crop = np.array(mask)[x:x+128,y:y+128]
                
            image_crop=Image.fromarray(image_crop) 
            mask_save_crop = np.array(mask_crop,dtype=np.uint8)
            mask_save_crop[mask_save_crop > 0] = 255
            mask_save_crop=Image.fromarray(mask_save_crop)

            save_path = path+str(i)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_image_path = save_path +'\\images\\'
            save_mask_path = save_path+'\\masks\\'
            save_whole_mask_path = save_path+'\\whole\\'
            if not os.path.exists(save_image_path):
                os.mkdir(save_image_path)
            if not os.path.exists(save_mask_path):
                os.mkdir(save_mask_path)
            if not os.path.exists(save_whole_mask_path):
                os.mkdir(save_whole_mask_path)
            image_crop.save(save_image_path + str(i) +".png", format='PNG')
            mask_save_crop.save(save_whole_mask_path + str(i) +".png", format='PNG')
            i += 1
            #select_masks(mask_crop,save_mask_path)  
def c_distance(center,feature_c):
    min_dist = np.max(pdist(np.array(feature_c), 'Euclidean'))
    c_dist = 0
    for f in feature_c:
        c_dist += np.sqrt(np.sum(np.square(f - center)))
    return c_dist/4+min_dist

def res_model(model,path_crop,c):  
    features = []
    #imgpath='/path/to/img.jpg'
    path = path_crop+str(c)+'/'
    length = len(os.listdir(path))
    print(length)
    for i in range(0,length,4):
        #batch = None
        for j in range(i,i+4):
            img_path = path + str(j) + '/images/'+str(j)+'.png'
                
            transform1 = transforms.Compose([
                transforms.Scale(128),
                transforms.ToTensor(),]
            )
            img = Image.open(img_path)
            img1 = transform1(img)
            img1 = img1.view(1,3,128,128)
            if j == i:
                batch = img1
            else:
                batch = torch.cat([batch,img1],0)
        #print(batch.size())
        feature = extract_feature(model, batch)
        #print(feature.shape)	# 打印出得到的tensor的shape
        feature = np.array(feature).reshape(feature.shape[0],2048)
        if i == 0:
            all_features = feature
        else:
            all_features = np.concatenate((all_features,feature),axis=0)
    return all_features
if __name__=="__main__":
    model=make_model()
    all_features = []
    path = 'data/'
    path_crop = 'data_30000_cut/'
    K1 = 6
    K2 = 4
    for i in range(0,30000,50):
        for j in range(i,i+50):
            img_path = path + str(j) + '/images/'+str(j)+'.png'
                
            transform1 = transforms.Compose([
                transforms.Scale(256),
                transforms.ToTensor(),]
            )
            img = Image.open(img_path)
            img1 = transform1(img)
            img1 = img1.view(1,3,256,256)
            if j == i:
                batch = img1
            else:
                batch = torch.cat([batch,img1],0)

        feature = extract_feature(model, batch)
        print(feature.shape)
        feature = np.array(feature).reshape(feature.shape[0],2048)
        if i == 0:
            all_features = feature
        else:
            all_features = np.concatenate((all_features,feature),axis=0)
    print(all_features.shape)
    
    ###first stage Kmeans clustering
    kmeans = KMeans(init="random",n_clusters=K1,n_init=10,max_iter=10000,random_state=42)
    kmeans.fit(all_features)
    p_labels = list(kmeans.labels_)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, all_features)
    closest = list(closest)
    
    ###crop patch into 4 128 patches
    for c in range(K1):
        indexs = [i for i in range(len(p_labels)) if p_labels[i] == c]
        i = 0
        for index in indexs:
            image = Image.open(path + str(index) + '\\images\\' + str(index) + '.png')
            mask = Image.open(path + str(index) + '\\whole\\' + str(index) + '.png')
            crop(image,mask,path_crop,c,i)
            i += 4
    model=make_model()
    results = []
    
    ###second stage clustering patch selection
    for c in range(K1):
        features = []
        features = res_model(model,path_crop,c)
        indexs = [i for i in range(len(p_labels)) if p_labels[i] == c]
        print(features.shape)
        kmeans_2 = KMeans(init="random",n_clusters=K2,n_init=10,max_iter=50000,random_state=500)
        kmeans_2.fit(features)
        centers = kmeans_2.cluster_centers_
        closest, _ = pairwise_distances_argmin_min(centers, features)
        closest = list(closest)
        c_labels = list(kmeans_2.labels_)
        counts = {}
        max_k = 0
        for k in range(K2):
            counts[k] = c_labels.count(k)
            if counts[k] > max_k:
                max_k = counts[k]
                keep = k
        c_indexs = [i for i in range(len(c_labels)) if c_labels[i] == keep]
        dist = []
        keep_index = []
        for i in c_indexs:
            j = i//4 * 4
            dist_c = np.sqrt(np.sum(np.square(all_features[indexs[i//4]] - kmeans.cluster_centers_[c])))
            dist.append(dist_c+c_distance(centers[keep],[features[j],features[j+1],features[j+2],features[j+3]]))
            keep_index.append(j)
        find_index = dist.index(min(dist))
    

        results.append(indexs[keep_index[find_index]//4])
    print('The selected patches are ',results)


# In[ ]:




