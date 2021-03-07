
import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
import pickle
def train(opt,Gs,Zs,reals,masks,real_masks,masked_reals,back_reals,NoiseAmp):

    in_s = 0
    scale_num = 0

    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        opt.outsave = opt.outf+'/fakesamples'
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass
        try:
            os.makedirs(opt.outsave)
        except OSError:
                pass
        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        G_curr = init_generator(opt)
        

        D_curr = init_discriminator(opt)
        D_fore = init_discriminator(opt)
        D_back = init_discriminator(opt)

        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))
            D_fore.load_state_dict(torch.load('%s/%d/netD_fore.pth' % (opt.out_,scale_num-1)))
            D_back.load_state_dict(torch.load('%s/%d/netD_back.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,D_fore,D_back,G_curr,reals,masked_reals,back_reals,masks,real_masks,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()
        D_fore = functions.reset_grads(D_fore,False)
        D_fore.eval()
        D_back = functions.reset_grads(D_back,False)
        D_back.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)

        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,D_fore,D_back,G_curr
    return



def train_single_scale(netD,netD_fore,netD_back,netG,reals,masked_reals,back_reals,masks,real_masks,Gs,Zs,in_s,NoiseAmp,opt,centers=None):
    path = opt.input_dir + 'randommasks/'
    files = os.listdir(path)
    real = reals[len(Gs)]
    masked_real = masked_reals[len(Gs)]
    back_real = back_reals[len(Gs)]
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD_fore = optim.Adam(netD_fore.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD_back = optim.Adam(netD_back.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerD_fore = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD_fore,milestones=[1600],gamma=opt.gamma)
    schedulerD_back = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD_back,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG_f2plot = []
    errG_b2plot = []
    err_back2plot = []
    D_real2plot = []
    D_fake2plot = []
    
    z_opt2plot = []
    err_fore2plot = []
    D_fore2plot = []
    D_mask2plot = []

    prevs = []


    real_mask = m_noise(real_masks[len(Gs)])
    
    for epoch in range(opt.niter):
        print('Epoch: ', epoch)
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)
        for i in range(len(masks)):
            mask = m_noise(masks[i][len(Gs)])
            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
            #print('Discriminator iteration ',j)
            # train with real
                netD.zero_grad()
                netD_fore.zero_grad()
                netD_back.zero_grad()

                output_real = netD(real).to(opt.device)
                print('The maximum value of output_real',torch.max(output_real))
                print('The minimum value of output_real',torch.min(output_real))
                #norm_mask = functions.norm_mask(masks[len(Gs)])
                errD_real = -output_real.mean()#-a
                errD_real.backward(retain_graph=True)
                #print('D real ',errD_real)
            
                ###D_fore process masked_real forehead of real
                output_real_fore = netD_fore(masked_real).to(opt.device)
                output_real_fore_masked = functions.upsampling_BW(functions.denorm(real_masks[len(Gs)]),output_real_fore.size()[2],output_real_fore.size()[3]) * output_real_fore
                #nums = torch.nonzero(output_real_fore_masked)

                #errD_real_fore = -output_real_fore_masked.sum()/(len(nums)+1)#-a
                errD_real_fore = -output_real_fore_masked.mean()
                errD_real_fore.backward(retain_graph=True)

                ###D_back process back_real background of real
                output_real_back = netD_back(back_real).to(opt.device)
                output_real_back_masked = functions.upsampling_BW((1-functions.denorm(real_masks[len(Gs)])),output_real_back.size()[2],output_real_back.size()[3]) * output_real_back
                #nums = torch.nonzero(output_real_back_masked)
                #errD_real_back = -output_real_back_masked.sum()/(len(nums)+1)#-a
                errD_real_back = -output_real_back_masked.mean()
                errD_real_back.backward(retain_graph=True)

                D_x = -errD_real.item()
                D_fore = -errD_real_fore.item()
                D_back = -errD_real_back.item()


                # train with fake
                if (j==0) & (epoch == 0):
                    if (Gs == []) & (opt.mode != 'SR_train'):
                        prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                        in_s = prev
                        prev = m_image(prev)
                        z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                        z_prev = m_noise(z_prev)
                        opt.noise_amp = 1
                    elif opt.mode == 'SR_train':
                        z_prev = in_s
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init * RMSE
                        z_prev = m_image(z_prev)
                        prev = z_prev
                    else:
                    
                        prev = draw_concat(Gs,Zs,reals,masks[i][:len(Gs)+1],real_masks,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                        prev = m_image(prev)
                        z_prev = draw_concat(Gs,Zs,reals,masks,real_masks,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init*RMSE
                        z_prev = m_image(z_prev)
                else:
                    prev = draw_concat(Gs,Zs,reals,masks[i][:len(Gs)+1],real_masks,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)

                if opt.mode == 'paint_train':
                    prev = functions.quant2centers(prev,centers)
                    plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)


                if (Gs == []) & (opt.mode != 'SR_train'):
                    noise = noise_
                else:
                    noise = opt.noise_amp*noise_+prev
                ##transfer fake to 1-D and get the crop areas
                concat = torch.cat((noise, mask), 1)
                fake = netG(concat.detach(),prev)
                output = netD(fake.detach())
                errD_fake = output.mean()
                
                ###masked_fore for fake image

                masked_fake = functions.masked(fake.detach(),masks[i][len(Gs)],opt)
                output_fake_fore = netD_fore(masked_fake)
                output_fake_fore_masked = functions.upsampling_BW(functions.denorm(masks[i][len(Gs)]),output_fake_fore.size()[2],output_fake_fore.size()[3]) * output_fake_fore
                errD_fake_fore = output_fake_fore_masked.mean()
                
                ###masked_back for fake image
                back_fake = functions.masked(fake.detach(),-masks[i][len(Gs)],opt)
                output_fake_back = netD_back(back_fake).to(opt.device)
                output_fake_back_masked = functions.upsampling_BW((1-functions.denorm(masks[i][len(Gs)])),output_fake_back.size()[2],output_fake_back.size()[3]) * output_fake_back
                errD_fake_back = output_fake_back_masked.mean() 
                gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
                gradient_penalty_fore = functions.calc_gradient_penalty(netD_fore, masked_real, masked_fake, opt.lambda_grad, opt.device)
                gradient_penalty_back = functions.calc_gradient_penalty(netD_back, back_real, back_fake, opt.lambda_grad, opt.device)

                #print(len(fakes))
            
            
                errD_fake.backward(retain_graph=True)         
                errD_fake_fore.backward(retain_graph=True)
                errD_fake_back.backward(retain_graph=True)
            
                D_G_z = errD_fake.item()
                D_G_fore = errD_fake_fore.item()
                D_G_back = errD_fake_back.item()

            
                gradient_penalty.backward() 
                gradient_penalty_fore.backward()
                gradient_penalty_back.backward()
            
                errD = errD_real + errD_fake + gradient_penalty
                err_fore = errD_real_fore + errD_fake_fore + gradient_penalty_fore
                err_back = errD_real_back + errD_fake_back + gradient_penalty_back
                optimizerD.step()
                optimizerD_fore.step()
                optimizerD_back.step()

            #errD2plot.append(errD.detach())
            #err_fore2plot.append(err_fore.detach())
            #err_back2plot.append(err_back.detach())
            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(opt.Gsteps):
                #print('Generator iteration ',j)
                netG.zero_grad()
                errG = 0
                errG_f = 0
                errG_b = 0
                output = netD(fake)
                errG = -output.mean()            
                output_fore = netD_fore(masked_fake)
                output_fore_masked = functions.upsampling_BW(functions.denorm(masks[i][len(Gs)]),output_fore.size()[2],output_fore.size()[3]) * output_fore
                errG_f = -output_fore_masked.mean()
                output_back = netD_back(back_fake)
                output_back_masked = functions.upsampling_BW((1-functions.denorm(masks[i][len(Gs)])),output_back.size()[2],output_back.size()[3]) * output_back
                errG_b = -output_back_masked.mean()
            
                errG.backward(retain_graph=True)         
                errG_f.backward(retain_graph=True)
                errG_b.backward(retain_graph=True)
            

            

            
                if alpha!=0:
                    loss = nn.MSELoss()
                    if opt.mode == 'paint_train':
                        z_prev = functions.quant2centers(z_prev, centers)
                        plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                    Z_opt = opt.noise_amp*z_opt+z_prev
                    concat = torch.cat((Z_opt, real_mask), 1)
                    rec_loss = alpha*loss(netG(concat.detach(),z_prev),real)
                    rec_loss.backward(retain_graph=True)
                    rec_loss = rec_loss.detach()
                else:
                    Z_opt = z_opt
                    rec_loss = 0

                optimizerG.step()
            print('error D_real: ',errD_real.detach())
            print('Error D: ',errG.detach())
            print('error real_fore:',errD_real_fore)
            print('error fore:',errG_f)
            print('error real_back:',errD_real_back)        
            print('error back:',errG_b)
            print('recon loss:',rec_loss)
        #errG_f2plot.append(errG_f.detach())
        #errG_b2plot.append(errG_b.detach())
            #D_real2plot.append(D_x)
            #D_mask2plot.append(D_fore)        
            #D_fake2plot.append(D_G_z)
            #D_fore2plot.append(D_G_fore)
            #z_opt2plot.append(rec_loss)

            if epoch % 25 == 0 or epoch == (opt.niter-1):
                print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

            if epoch % 100 == 0 or epoch == (opt.niter-1):
                plt.imsave(opt.outsave + '/'+files[i], functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
                #print(functions.convert_image_np(fake_gray.detach()).shape)
                #plt.imsave('%s/fake_crop.png' %  (opt.outf), functions.convert_image_np(masked_fake.detach()),vmin=0, vmax=1)
                #plt.imsave('%s/fake_crop_1.png' %  (opt.outf), functions.convert_image_np(masked_fake.detach()),cmap='jet',vmin=0,vmax=1)
                #plt.imsave('%s/random_mask.png' %  (opt.outf), functions.convert_image_np(masks[len(Gs)].detach()),vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(concat.detach(), z_prev).detach()), vmin=0, vmax=1)
                plt.imsave('%s/output.png'   % (opt.outf), functions.convert_image_np(output.detach()),cmap='jet')
                plt.imsave('%s/output_real.png'   % (opt.outf), functions.convert_image_np(output_real.detach()),cmap='jet')
                plt.imsave('%s/output_real_fore_masked.png'   % (opt.outf), functions.convert_image_np(output_real_fore_masked.detach()),cmap='jet')
                plt.imsave('%s/output_real_back_masked.png'   % (opt.outf), functions.convert_image_np(output_real_back_masked.detach()),cmap='jet')
                plt.imsave('%s/output_real_fore.png'   % (opt.outf), functions.convert_image_np(output_real_fore.detach()),cmap='jet')
                plt.imsave('%s/output_real_back.png'   % (opt.outf), functions.convert_image_np(output_real_back.detach()),cmap='jet')
                plt.imsave('%s/output_fake_fore.png'   % (opt.outf), functions.convert_image_np(output_fake_fore.detach()),cmap='jet')
                plt.imsave('%s/output_fake_back.png'   % (opt.outf), functions.convert_image_np(output_fake_back.detach()),cmap='jet')
                plt.imsave('%s/output_fake_fore_masked.png'   % (opt.outf), functions.convert_image_np(output_fake_fore_masked.detach()),cmap='jet')
                plt.imsave('%s/output_fake_back_masked.png'   % (opt.outf), functions.convert_image_np(output_fake_back_masked.detach()),cmap='jet')
                #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
                #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
                plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
                #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
                plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


                torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()
        schedulerD_fore.step()
        schedulerD_back.step()
    '''
    file = opt.outf + '/plot.pickle'
    with open(file,'wb') as file:
        pickle.dump(
        {
            'errG_f2plot':errG_f2plot,
            'errG_b2plot':errG_b2plot,
            'errD2plot':errD2plot,
            'err_back2plot':err_back2plot,
            'err_fore2plot':err_fore2plot,
            'D_real2plot':D_real2plot,
            'D_mask2plot':D_mask2plot,
            'D_fake2plot':D_fake2plot,
            'D_fore2plot':D_fore2plot,
            'z_opt2plot':z_opt2plot,
        },
        file,pickle.HIGHEST_PROTOCOL)  
    '''        
    functions.save_networks(netG,netD,netD_fore,netD_back,z_opt,opt)
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,masks,real_masks,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    prevs = []
    G_z = in_s
    
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,mask,noise_amp in zip(Gs,Zs,reals,reals[1:],masks,NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    #z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = torch.full([1,opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], 0, device=opt.device)
                z = m_noise(z)
                mask = m_noise(mask)
                #print(mask.size())
                #print(z.size())
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                concat = torch.cat((z_in, mask), 1)
                G_z = G(concat.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,mask,noise_amp in zip(Gs,Zs,reals,reals[1:],real_masks,NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                mask = m_noise(mask)
                z_in = noise_amp*Z_opt+G_z
                concat = torch.cat((z_in, mask), 1)
                G_z = G(concat.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_generator(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    #print(netG)
    return netG
def init_discriminator(opt):
    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)

    return netD
def init_maskdiscriminator(opt):
    #discriminator initialization:
    netD = models.MDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)

    return netD