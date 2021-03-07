import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ConvBlock_D(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock_D,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('Relu6',nn.ReLU6())

class ConvBlock_G(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock_G,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def getheatmap(feature,dst,block):
    iter_range = feature.shape[1]
    feature = feature.cpu().data.numpy()
    for i in range(iter_range):        
        
        feature_img = feature[:,i,:,:]
        #print(feature_img.shape)
        feature_img = feature_img.reshape(feature_img.shape[1],feature_img.shape[2],1)
        #feature_img = np.asarray()
        feature_img = 255 * feature_img
        feature_img = feature_img.astype(np.uint8)    
        dst_path = os.path.join(dst,block)
        #print(dst_path)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        #print(feature_img.dtype,feature_img.shape)
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        dst_file = os.path.join(dst_path, str(i) + '.png')
        #print(dst_file)
        cv2.imwrite(dst_file, feature_img)
        
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock_D(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock_D(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        #self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
        #self.tail = ConvBlock(max(N,opt.min_nfc),1,opt.ker_size,opt.padd_size,1)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.ReLU6()
        )


    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
        
class MDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        m = nn.LayerNorm(x.size()[1:],eps=0, elementwise_affine=False)
        x = m(x)
        return x

class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock_G(4,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock_G(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y
