from sys import modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import matplotlib
import matplotlib.image as mpimg

class Ploter():
    
    def __init__(self,rpath="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/",**kwargs):
        self.rpath = rpath
        # self.backbones = self._default_para(kwargs,"backbone",["bs10_unet","bs10_resnet","bs10_unet_128_basic","bs10_aruppp","bs10_arupp"])
        self.backbones =["bs10_unet","bs10_resnet","bs10_unet_128_basic","bs10_denseunet","bs10_unet_128","bs10_sarumm","bs10_sarumm_basic"]
    def _default_para(self,dict,para,defult_input):
        if para not in dict.keys():
            return defult_input
        else:
            return dict[para]

    def _get_pic_list(self,pic_name):
        self.pics = {}
        for backbone in self.backbones:
            pic_path = os.path.join(self.rpath,backbone,"test_best/images/")+pic_name
            self.pics[backbone] = self._get_np(pic_path)
        return 0
    
    def _get_np(self,pic_path):
        return mpimg.imread(pic_path)
    
    def plot(self,picname):
        pic_name = picname + "_fake_B.png"
        self._get_pic_list(pic_name)
        
        self.pics["real_A"] = self._get_np(os.path.join(self.rpath,"bs10_arupp","test_best/images/")+picname+ "_real_A.png")
        plt.figure(dpi=200)
        plt.imshow(self.pics["real_A"])
        plt.title("MRI")
        plt.axis('off')
        plt.show()
        
        self.pics["real"] = self._get_np(os.path.join(self.rpath,"bs10_arupp","test_best/images/")+picname+ "_real_B.png")
        plt.figure(dpi=200)
        plt.imshow(self.pics["real"])
        plt.title("Real CT")
        plt.axis('off')
        plt.show()
        
        

        for backbone in self.backbones:
            plt.figure(dpi=200)
            plt.imshow(self.pics[backbone])
            plt.title(backbone)
            plt.axis('off')
            plt.show()


