from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    masked_reals = []
    back_reals = []
    ring_reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real_ = functions.read_image(opt)
        functions.adjust_scales2image(real_, opt)
        ###Get real images in different size
        
        real = imresize(real_,opt.scale1,opt)
        reals = functions.creat_reals_pyramid(real,reals,opt)
        
        ###Get random masks
        
        masks = functions.get_masks(opt,reals)

        ###Get masks for train image
        real_masks = functions.get_real_masks(opt,reals)


        ###mask on forehead
        masked_real_ = functions.read_image_mask(opt.input_name,opt.mask_name,'fore',opt)
        masked_real = imresize(masked_real_,opt.scale1,opt)
        masked_reals = functions.creat_reals_pyramid(masked_real,masked_reals,opt) ##set of masked real images for different scales
    
    
        ###mask on background
        back_real_ = functions.read_image_mask(opt.input_name,opt.mask_name,'back',opt)
        back_real = imresize(back_real_,opt.scale1,opt)
        back_reals = functions.creat_reals_pyramid(back_real,back_reals,opt)
    
    

        train(opt, Gs, Zs, reals,masks,real_masks,masked_reals,back_reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
