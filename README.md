# Label-Effcient-Nuclei-Segmentation
Official code implementation for paper 'Automatic Patch Selection for Label-Effcient Nuclei Segmentation'

The whole framework consists of 3 stages: 1. **patch selection** 2. **CSinGAN data augmentation** 3. **Pseudo-label based MRCNN**

## Stage 1: Consistency-based Patch Selection

### **Data**
The dataset that we used in this paper is the Monusg dataset: https://monuseg.grand-challenge.org/Data/.

In first-stage clustering, the 1000\*1000 images are cropped into 256\*256 patches and stored in '*data/*'.

In second-stage clustering, the 256\*256 patches are cropped into 128\*128 patches and stored in '*data_30000_cut*'.

The selected results are the folder names in '*data/*'. The tunable parameters are K1, K2 which are the clutser numbers for 2 stages.

### **Code**
python consis_based.py

## Stage 2: CSinGAN augmentation
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
