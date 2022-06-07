# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝

# @Time    : 2021-09-28 09:35:49
# @Author  : zhaosheng
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : iint.icu
# @File    : models/generate_model.py
# @Describe: For pix2pix models.


from util.ct_loss import PerceptualLoss
import torch
from .base_model import BaseModel
from . import networks
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import os
from PIL import Image

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        
        parser.set_defaults(norm='batch', dataset_mode='aligned') #  netG='cbamunet',
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.dream = opt.dream
        if opt.style_loss:
            self.loss_names = ['G_L1']#,'G_GAN',  'D_real', 'D_fake','Style']
        else:
            self.loss_names = ['G_L1']#,'G_GAN',  'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        print(self.netG)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.content_loss = PerceptualLoss(torch.nn.MSELoss())
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, 'min', patience=10, factor=0.5, min_lr=0.0000002)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        #print(f"Fake B shape:{self.fake_B.shape}")
        

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminators
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) 
        self.loss_G_L1 = torch.abs(self.loss_G_L1 - self.dream) + self.dream
        # combine loss and calculate gradients
        if self.opt.style_loss:
            self.loss_Style = self.content_loss.get_loss(torch.cat([self.fake_B,self.fake_B,self.fake_B], dim=1), torch.cat([self.real_B,self.real_B,self.real_B], dim=1))
            #self.loss_CT = self.criterionCT(self.fake_B, self.real_B)
            self.loss_G = self.loss_G_GAN + self.loss_G_L1* self.opt.lambda_L1 + self.loss_Style
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1* self.opt.lambda_L1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def change_lr(self,socre):
        self.scheduler.step(socre)
        
    def get_test_img_array(input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.
    
        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def get_test_score(self,opt,loader,visualizer,pic_name,win_id,save_png):
        self.netG.eval()
        if save_png:
            os.makedirs(f"./results/{opt.name}/temp_pics/k{opt.k_index}_{opt.now_epoch}/",exist_ok=True)

        n_val = len(loader)  # the number of batch
        tot = 0
        _n = 0
        criterion_pixelwise = torch.nn.L1Loss()
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for batch in loader:
                self.real_A = batch['A'].to(self.device)
                self.real_B = batch['B'].to(self.device)
                self.image_paths = batch['A_paths']
                with torch.no_grad():
                    self.fake_B = self.netG(self.real_A)
                    if opt.display_val_test and opt.display_id > 0:
                        visual_ret = OrderedDict()
                        for name in ['real_A', 'fake_B', 'real_B']:
                            visual_ret[name] = getattr(self, name)
                        visualizer.display_current_results(visual_ret, opt.now_epoch, True,win_id,pic_name)

                for xx in range(self.fake_B.shape[0]):
                    pic_file_name = batch['A_paths'][xx].split(".")[0]
                    fake_ct = self.fake_B[xx,0,:,:]
                    real_ct = self.real_B[xx,0,:,:]
                    _loss = criterion_pixelwise((fake_ct+1)/2.0*1800-1000,(real_ct+1)/2.0*1800-1000 ).item()
                    tot += _loss
                    _n += 1

                    if save_png:
                        # [-1,1] -> [0,255] -> Image -> png
                        real_ct_array = np.tile(real_ct.data.cpu().float().numpy(), (3, 1, 1))
                        fake_ct_array = np.tile(fake_ct.data.cpu().float().numpy(), (3, 1, 1))
                        fake_ct_array = (np.transpose(fake_ct_array, (1, 2, 0)) + 1) / 2.0 * 255.0
                        real_ct_array = (np.transpose(real_ct_array, (1, 2, 0)) + 1) / 2.0 * 255.0
                        real_ct_im = Image.fromarray(real_ct_array.astype(np.uint8)).resize((256,256), Image.BICUBIC)#.convert('L')
                        fake_ct_im = Image.fromarray(fake_ct_array.astype(np.uint8)).resize((256,256), Image.BICUBIC)#.convert('L')
                        real_ct_im.save(f"./results/{opt.name}/temp_pics/k{opt.k_index}_{opt.now_epoch}/{pic_file_name}_real_B.png")
                        fake_ct_im.save(f"./results/{opt.name}/temp_pics/k{opt.k_index}_{opt.now_epoch}/{pic_file_name}_fake_B.png")    
                pbar.update(self.fake_B.shape[0])
        self.netG.train()
        return tot / _n
