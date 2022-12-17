# Label-Effcient-Nuclei-Segmentation
Official code implementation for paper 'Which Pixel to Annotate: a Label-Efficient Nuclei Segmentation Framework'

The whole framework consists of 3 stages: 1. **Patch selection** 2. **CSinGAN data augmentation** 3. **Pseudo-label based MRCNN**

## Stage 1: Consistency-based Patch Selection
In this stage, patches with representativity and consistency will be selected for data augmentation and semi-supervised segmentation.

### **Data**
The dataset that we used in this paper is the Monusg dataset: https://monuseg.grand-challenge.org/Data/.

In first-stage clustering, the 1000\*1000 images are cropped into 256\*256 patches and stored in '*data/*'.

In second-stage clustering, the 256\*256 patches are cropped into 128\*128 patches and stored in '*data_30000_cut*'.

The selected results are the folder names in '*data/*'. The tunable parameters are K1, K2 which are the clutser numbers for 2 stages.

### **Code**
python consis_based.py

## Stage 2: CSinGAN augmentation
In stage 2, the random masks are synthesized by randomly permuting the nuclei positions in the selected patches' mask. Then the real pair and random masks are the inputs for CSinGAN.

### **Training Data**
**train.png**: the selected real patch

**train_mask.png**: the corresponding mask for the real patch

**randon mask/**: the synthesized masks that we want to add patterns

### **Code**
### **Dependencies**
torch==1.4.0; torchvision==0.5.0 ; opencv

### **Training**
python main_train.py

## Stage 3: Pseudo-label based MRCNN
This Mask-rcnn code implementation is modified by Matterport's implementation of Mask-RCNN: https://github.com/matterport/Mask_RCNN. The evaluation code is modified for pseudo label prediction. The environment dependencies and data strcture are following the same setting with Matterport's codes.

### **Pseudo-label.py**
This code is to predict pseudo label for cropped image patches to perform semi-supervised learning. The pretrained model is trained by the synthesis images created by CSinGAN. And the unlabelled images with predicted pseudo labels can form new image paires for new MRCNN training. 


## Citation
If you find this research useful in your work, please acknowledge it appropriately and cite the paper:
```bibtex
@ARTICLE{9946007,
  author={Lou, Wei and Li, Haofeng and Li, Guanbin and Han, Xiaoguang and Wan, Xiang},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Which Pixel to Annotate: a Label-Efficient Nuclei Segmentation Framework}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3221666}}
```
