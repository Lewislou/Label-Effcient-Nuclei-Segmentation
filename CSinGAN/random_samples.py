from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import PIL.Image as Image

def random_sample(dir2save,mask_dir,input_name):
    parser = get_arguments()
    #parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    #parser.add_argument('--input_name', help='input image name',default=input_name)
    parser.add_argument('--mask_name', help='input mask name',default=mask_dir+input_name)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='random_samples')
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    if dir2save is None:
        print('task does not exist')
    else:
        if opt.mode == 'random_samples':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,1,1,opt)
            out = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale)

        elif opt.mode == 'random_samples_arbitrary_sizes':
            real = functions.read_image(opt)
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            in_s = functions.generate_in2coarsest(reals,opt.scale_v,opt.scale_h,opt)
            out = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, in_s, scale_v=opt.scale_v, scale_h=opt.scale_h)
        array = functions.convert_image_np(out.detach())*255
        Img = Image.fromarray(array.astype(np.uint8))
            #lt.imsave(dir2save + ref_name, functions.convert_image_np(out.detach()), vmin=0, vmax=1)
        Img.save(dir2save + input_name)

dir2save ='samples//'
mask_dir = 'masks//'
for file in os.listdir(mask_dir):
    random_sample(dir2save,mask_dir,file)


