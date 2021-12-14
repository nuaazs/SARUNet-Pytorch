import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import wandb
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import matplotlib.pyplot as plt

from cbam_models.SmaAt_UNet import SmaAt_UNet
from cbam_models.layers import CBAM
import torch
import numpy as np
from PIL import Image


if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)

    model = SmaAt_UNet(1, 1)
    load_path = opt.checkpoint
    
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)

    print(model)
    target_layers = [model.cbam3.spatial_att.conv4]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        grayscale_cam = cam(input_tensor=data['A'],target_category=None)
        grayscale_cam = grayscale_cam[0, :]
        print(grayscale_cam.shape)
        pic_array = np.hstack([np.array(grayscale_cam)*255.,np.array(data['A'].cpu()[0,0,:,:])*255.])
        print(pic_array.shape)
        # plt.figure()
        # plt.imshow(np.array(grayscale_cam),cmap='heat')
        # plt.savefig(f"./cam_pngs/{i}_plt.png")
        # #plt.show()

        # plt.figure()
        # plt.imshow(np.array(data['A'].cpu()[0,0,:,:]),cmap='gray')
        # plt.savefig(f"./cam_pngs/{i}_plt.png")
        # #plt.show()

        assert(pic_array.shape==(256, 512))
        img = Image.fromarray(pic_array)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(f"./cam_pngs/{i}.png")