from sys import modules
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import matplotlib
import matplotlib.image as mpimg

class Ploter():
    
    def __init__(self,rpath,**kwargs):
        self.rpath = rpath
        self.backbones = self._default_para(kwargs,"backbone",["unet","resnet","attnunet"])

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
    
    def plot(self,pic_name):
        self._get_pic_list(pic_name)
        plt.figure(dpi=400)
        for backbone in self.backbones:
            plt.imshow(self.pics[backbone])
        plt.title(backbone)
        plt.show()
        return 0


