import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize
import os
import random
from sklearn.cluster import KMeans
import cv2

# custom weights initialization called on netG and netD

def read_image(opt):
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    return np2torch(x)
    
def read_image_mask(image_path,mask_path,label,opt):
    x = img.imread(opt.input_dir+image_path)
    x = np2torch(x,'image',opt)
    x = x[:,0:3,:,:]
    if label == 'ring':
        mask = img.imread(opt.ring_dir+mask_path)
    else:
        mask = img.imread(opt.input_dir+mask_path)
    mask = np2torch(mask,'mask',opt)
    mask = mask[:,0:1,:,:]
    origin1 = origin(x,opt)
    if label == 'back':
        masked_origin = masked(origin1,-mask,opt)
    else:
        masked_origin = masked(origin1,mask,opt)
    plt.imsave('masked_'+label+image_path,torch2uint8(masked_origin))
    return masked_origin
def origin(x,opt):
    x = torch2uint8(x)
    x = np2torch(x,'image',opt)
    x = x[:,0:3,:,:]
    return x    
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)
def norm_mask(x):
    #out = (x -0.5) *2
    return x.clamp(-1, -0.85)
#def denorm2image(I1,I2):
#    out = (I1-I1.mean())/(I1.max()-I1.min())
#    out = out*(I2.max()-I2.min())+I2.mean()
#    return out#.clamp(I2.min(), I2.max())

#def norm2image(I1,I2):
#    out = (I1-I2.mean())*2
#    return out#.clamp(I2.min(), I2.max())

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
        inp = np.clip(inp,0,1)
    else:
        #inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        #inp = np.clip(inp,0,1)
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
   
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise
def get_masks(opt,reals):
    numbers = len(reals)
    masks = []
    temp_reals = []
    path = opt.input_dir + 'randommasks/'
    files = os.listdir(path)
    for file in files:
        mask = read_random_mask(path+file,opt)
        #mask = np2torch(mask,'mask',opt)
        curr_masks = []
        for i in range(numbers):
            opt.nzx = reals[i].shape[2]#+(opt.ker_size-1)*(opt.num_layer)
            opt.nzy = reals[i].shape[3]
            mask = upsampling_BW(mask,opt.nzx , opt.nzy)
            curr_masks.append(mask)

        masks.append(curr_masks)
        #temp_reals.append(temp_real)
    #save_masks('randommasks',masks)    
    return masks
def get_diff_ring_masks(opt,reals):
    numbers = len(reals)
    masks = []
    temp_reals = []
    for i in range(numbers):
        opt.nzx = reals[i].shape[2]#+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = reals[i].shape[3]
        mask= generate_mask([opt.nzx,opt.nzy],opt,i,device=opt.device,type='diff_ring')

        masks.append(mask)
        #temp_reals.append(temp_real)
    save_masks('diffringmasks',masks)    
    return masks 
def save_masks(dst_path,feature):
    iter_range = len(feature)
    
    for i in range(iter_range):                
        mask = denorm(feature[i])
        #mask = mask[:,0:1,:,:]
        mask = mask.cpu().numpy()*255
        mask = mask.reshape(mask.shape[2],mask.shape[3],1)
        mask = mask.astype(np.uint8)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        #print(feature_img.dtype,feature_img.shape)
        #feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        dst_file = os.path.join(dst_path, str(i) + '.png')
        #plt.imsave(dst_file,mask,cmap='gray')
        cv2.imwrite(dst_file, mask)  
        
def save_random_masks(dst_path,level,feature):              
    mask = denorm(feature)
    #mask = mask[:,0:1,:,:]
    mask = mask.cpu().numpy()*255
    mask = mask.reshape(mask.shape[2],mask.shape[3],1)
    mask = mask.astype(np.uint8)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        #print(feature_img.dtype,feature_img.shape)
        #feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
    dst_file = os.path.join(dst_path, str(level) + '.png')
        #plt.imsave(dst_file,mask,cmap='gray')
    cv2.imwrite(dst_file, mask)    
        
def get_real_masks(opt,reals):
    numbers = len(reals)
    masks = []
    temp_reals = []
    for i in range(numbers):
        opt.nzx = reals[i].shape[2]#+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = reals[i].shape[3]
        mask = generate_mask([opt.nzx,opt.nzy],opt,i,device=opt.device,type='train')

        masks.append(mask)
        #temp_reals.append(temp_real)
    #save_masks('realmasks',masks)
    return masks
def get_ring_masks(opt,reals):
    numbers = len(reals)
    masks = []
    temp_reals = []
    for i in range(numbers):
        opt.nzx = reals[i].shape[2]#+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = reals[i].shape[3]
        mask = generate_mask([opt.nzx,opt.nzy],opt,i,device=opt.device,type='ring')

        masks.append(mask)
        #temp_reals.append(temp_real)
    #save_masks('ringmasks',masks)
    return masks

def create_ring_for_mask(opt):
    path = opt.input_dir + '.\\randommasks\\'
    dirs = os.listdir(path)
    for file in dirs:
        mask2ring(path,file,opt)
def mask2ring(path,file,opt):
    mask_image = path + file
    img = cv2.imread(mask_image,0)
    contours , hierarchy = cv2.findContours ( img ,  cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE )
    new = np.zeros((img.shape[0],img.shape[1]))
    cv2.drawContours(new,contours,-1,(255,255,255),1)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(new,kernel,iterations = 1)
    ret, thresh = cv2.threshold(dilation, 0, 255,cv2.THRESH_BINARY)
    cv2.imwrite(opt.ring_dir+file, thresh)


def generate_mask(size,opt,level,device='cuda',type='diff'):
    if type == 'same':
        mask = read_same_mask(opt)
        #print(mask_.size())
        #mask = imresize_to_shape(mask_,size,opt)
        #real = imresize_to_shape(real_,size,opt)
        #noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        mask = upsampling_BW(mask,size[0], size[1])
        return mask
    '''    
    if type =='diff':
        masks = []
        path = 'randommask\\'
        for file in os.listdir(path):
        mask = read_mask(path+file)
        mask = upsampling_BW(mask,size[0], size[1])
        masks.append(mask)
        return masks
    '''
    if type =='train':
        mask = read_mask(opt)
        #mask = imresize_to_shape(mask_,size,opt)
        mask = upsampling_BW(mask,size[0], size[1])
        #print(mask.size())
        return mask
    if type =='ring':
        mask = read_ring_mask(opt)
        #mask = imresize_to_shape(mask_,size,opt)
        mask = upsampling_BW(mask,size[0], size[1])
        #print(mask.size())
        return mask
    if type =='diff_ring':
        mask = read_diff_ring_mask(level,opt)
        #mask = imresize_to_shape(mask_,size,opt)
        mask = upsampling_BW(mask,size[0], size[1])
        #print(mask.size())
        return mask
def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)
def upsampling_BW(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='nearest')
    return m(im)
def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def read_image(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = np2torch(x,'image',opt)
    x = x[:,0:3,:,:]
    return x
def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,'image',opt)
    x = x[:,0:3,:,:]
    return x
def read_mask(opt):
    x = img.imread(opt.mask_name)
    x = np2torch(x,'mask',opt)
    x = x[:,0:1,:,:]
    return x
def read_ring_mask(opt):
    mask2ring(opt.input_dir,'cell_train_mask.png',opt)
    x = img.imread('%s/%s' % (opt.ring_dir,'cell_train_mask.png'))
    x = np2torch(x,'mask',opt)
    x = x[:,0:1,:,:]
    return x
def read_diff_ring_mask(level,opt):
    x = img.imread('%s/%s' % (opt.ring_dir,str(level)+'.png'))
    x = np2torch(x,'mask',opt)
    x = x[:,0:1,:,:]
    return x
def read_same_mask(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.same_mask_name))
    x = np2torch(x,'mask',opt)
    x = x[:,0:1,:,:]
    return x
def read_random_mask(path,opt):
    x = np.array(img.imread(path))
    #print(image.shape)  
    x = np2torch(x,'mask',opt)

    x = x[:,0:1,:,:]

    return x
def np2torch(x,mode,opt):
    if mode == 'image':
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
        x = torch.from_numpy(x)
        if not(opt.not_cuda):
            x = move_to_gpu(x)
        x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
        x = norm(x)
    elif mode == 'mask':
        #x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
        x = torch.from_numpy(x)
        if not(opt.not_cuda):
            x = move_to_gpu(x)
        x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
        x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x
'''
def rgb2gray(x,mask,opt):
    x = torch2uint8(x)
    x = color.rgb2gray(x)
    mask = mask.cpu().numpy()
    #print(mask.shape)
    mask = mask.reshape(mask.shape[2],mask.shape[3])
    x = x*mask
    x = np2torch(x,'mask',opt)
    return x

def rgb2gray(x,mask,opt):
    x = torch2uint8(x)
    mask = torch2uint8(mask)
    mask = mask.reshape(mask.shape[0],mask.shape[1])
    x = color.rgb2gray(x)
    x = x*mask/255
    x = np2torch(x,'mask',opt)
    return x
'''
def masked(x,mask,opt):
    #print(mask.size())
    x = torch2uint8(x)
    mask = torch2uint8(mask)
    mask = mask.reshape(mask.shape[0],mask.shape[1])
    masked_fake = cv2.add(x, np.zeros(np.shape(x), dtype=np.uint8), mask=mask) 
    #rint(masked_fake.shape)
    masked_fake = np2torch(masked_fake,'image',opt)
    masked_fake = masked_fake[:,0:3,:,:]
    return masked_fake
def read_image2np(opt):
    x = img.imread('%s/%s' % (opt.input_dir,opt.input_name))
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,netD_fore,netD_back,z,opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(netD_fore.state_dict(), '%s/netD_fore.pth' % (opt.outf))
    torch.save(netD_back.state_dict(), '%s/netD_back.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    print('first round')
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    print('num_scales',opt.num_scales)
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    print('scale2stop',scale2stop)
    opt.stop_scale = opt.num_scales - scale2stop
    print('stop_scale',opt.stop_scale)
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    print('scale1',opt.scale1)
    print('second round')
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def adjust_scales2image_SR(real_,opt):
    opt.min_size = 18
    opt.num_scales = int((math.log(opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
    scale2stop = int(math.log(min(opt.max_size , max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    scale2stop = int(math.log(min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def creat_reals_pyramid(real,reals,opt):
    real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals
def creat_random_pyramid(mask,reals,masks,opt):
    mask = mask[:,0:1,:,:]
    for i in range(0,opt.stop_scale+1,1):
        size = reals[i].size()
        curr_mask = upsampling_BW(mask,size[2],size[3])
        masks.append(curr_mask)
        #print('current size',curr_mask.size())
        #print('current length:',len(masks))
    return masks

def load_trained_pyramid(opt, mode_='train'):
    #dir = 'TrainedModels/%s/scale_factor=%f' % (opt.input_name[:-4], opt.scale_factor_init)
    mode = opt.mode
    opt.mode = 'train'
    if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
        opt.mode = mode
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load('%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    real = reals[opt.gen_start_scale]
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    if opt.gen_start_scale == 0:
        in_s = torch.full(real_down.shape, 0, device=opt.device)
    else: #if n!=0
        in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.alpha)
    elif (opt.mode == 'animation_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (opt.input_name[:-4], opt.scale_factor_init)
    elif (opt.mode == 'paint_train') :
        dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (opt.input_name[:-4], opt.scale_factor_init,opt.paint_start_scale)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out,opt.input_name[:-4], opt.gen_start_scale)
    elif opt.mode == 'random_samples_arbitrary_sizes':
        dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
    elif opt.mode == 'animation':
        dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
    elif opt.mode == 'SR':
        dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
    elif opt.mode == 'harmonization':
        dir2save = '%s/Harmonization/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'editing':
        dir2save = '%s/Editing/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
    elif opt.mode == 'paint2image':
        dir2save = '%s/Paint2image/%s/%s_out' % (opt.out, opt.input_name[:-4],opt.ref_name[:-4])
        if opt.quantization_flag:
            dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    if opt.mode == 'SR':
        opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num

def quant(prev,device):
    arr = prev.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor.to(device))
    x = x.view(prev.shape)
    return x,centers

def quant2centers(paint, centers):
    arr = paint.reshape((-1, 3)).cpu()
    kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
    labels = kmeans.labels_
    #centers = kmeans.cluster_centers_
    x = centers[labels]
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    #x = x.type(torch.cuda.FloatTensor)
    x = x.view(paint.shape)
    return x

    return paint


def dilate_mask(mask,opt):
    if opt.mode == "harmonization":
        element = morphology.disk(radius=7)
    if opt.mode == "editing":
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:,:,0]
    mask = morphology.binary_dilation(mask,selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask,opt)
    opt.nc_im = nc_im
    mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
    plt.imsave('%s/%s_mask_dilated.png' % (opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0,vmax=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask


