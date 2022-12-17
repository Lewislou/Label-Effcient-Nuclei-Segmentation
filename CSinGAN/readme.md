# Implementation of our conditional single image GAN (CSinGAN)

## Getting started
## **Training Data**
**train.png**: the selected real patch

**train_mask.png**: the corresponding mask for the real patch

**randon mask/**: the synthesized masks that we want to add patterns

## **Code**
### **Dependencies**
torch==1.4.0; torchvision==0.5.0 ; opencv

### **Mask_synthesis**
python mask_synthesis.py

### **Training**
python main_train.py
