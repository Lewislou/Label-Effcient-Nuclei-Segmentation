#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Process whole slide images
#import openslide
from PIL import Image
import cv2
import skimage
from skimage.io import imread
import imageio
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import scipy
import scipy.ndimage
import scipy.io as scio
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import train
import tensorflow as tf


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        #assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)


        image_ids = [i for i in range(240)]


        # Add images
        if subset == "stage1_test":

            for image_id in image_ids:
                self.add_image(
                    "nucleus",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id, "images/{}.tif".format(image_id)))
        else:
            for image_id in image_ids:
                self.add_image(
                    "nucleus",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, "{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
VAL_IMAGE_IDS = [
    'TCGA-A7-A13E-01Z-00-DX1',
    'TCGA-A7-A13F-01Z-00-DX1',
    'TCGA-AR-A1AK-01Z-00-DX1',
    'TCGA-AR-A1AS-01Z-00-DX1',
    'TCGA-AY-A8YK-01A-01-TS1',
    'TCGA-B0-5698-01Z-00-DX1'
]
dataset_dir = '16images\\' ##path for cropped images
test_dataset = NucleusDataset()
test_dataset.load_nucleus(dataset_dir,'images')
test_dataset.prepare()
print("Images: {}\nClasses: {}".format(len(test_dataset.image_ids), test_dataset.class_names))

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

    #dataNew = name + '_predicated.mat'



# In[2]:


############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus
    STEPS_PER_EPOCH = 40
    VALIDATION_STEPS = 1

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
#     MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([47.49, 41.63, 51.28])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 500

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 500
#     MAX_GT_INSTANCES = 2000

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 1000
#     DETECTION_MAX_INSTANCES = 2000


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    #scio.savemat(dataNew, {'predicted':cropped})
config = NucleusInferenceConfig()
config.display()


# In[3]:


import scipy.io as scio
import os.path as osp
import shutil
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
path = 'models/'
with tf.device(DEVICE):
    nmodel = modellib.MaskRCNN(mode="inference",
                            model_dir=os.getcwd(),
                            config=config)
weights_path = "/usingfakemask/mask_rcnn_nucleus.h5" ###trained model weight
nmodel.load_weights(weights_path, by_name=True)

def compute_overlaps_masks(masks1, masks2):
    '''
    masks1, masks2: [Height, Width, instances]
    '''
    
    # If either set of masks is empty return empty result
    #print(masks1.shape,masks2.shape)
    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)
 
    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    #print(overlaps)
    return overlaps
def save(mask,i,path,dirs):
    mask_path = path + dirs
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    mask.save(osp.join(mask_path,str(i)+".png"))
def create_mask(image,save_path,nmodel):
    results = nmodel.detect([image], verbose=1)
    r = results[0]
    # Generate an n-ary nuclei mask 
    individual_nuclei = r['masks']
    predicted_nuclei =np.zeros((256,256), dtype = int)
    n_nuc = 0
    for k in range(individual_nuclei.shape[2]):
        n_nuc += 1
        nuc_mask = r['masks'][:,:,k]
        predicted_nuclei += (n_nuc)*nuc_mask
        save(Image.fromarray(nuc_mask),k,save_path,'masks')
        
    new_nuclei = np.zeros((256,256), dtype = int)

    rows,cols=new_nuclei.shape
    for i in range(rows):
        for j in range(cols):
            if (predicted_nuclei[i,j]<=0):
                new_nuclei[i,j]=0
            else:
                new_nuclei[i,j]=1
    whole_path = save_path+'whole\\'
    os.mkdir(whole_path)
    skimage.io.imsave(whole_path+'0.png',new_nuclei)
        

path = "16images\\images\\" ## images path for 240 cropped images
new_path = "16images\\pseudo_center_2400\\" ##save new images and labels
for i in range(240):
    #image = test_dataset.load_image(i)

    image = test_dataset.load_image(i)
    info = test_dataset.image_info[i]
    file = info["id"]
    save_path = new_path + str(file)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.mkdir(save_path+'\\images\\')
    shutil.copy(path+str(file)+'.png', save_path+'\\images\\cell.png')
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    
    create_mask(image,save_path+'\\',nmodel)





