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
# @File    : data/aligned_dataset.py
# @Describe:

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, get_mat
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import ants
from torch.autograd import Variable

import random


class AlignedDataset_img(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "mr")
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "ct")
        self.A_paths = sorted(
            [_file for _file in os.listdir(self.dir_A) if "png" in _file])
        self.B_paths = sorted(
            [_file for _file in os.listdir(self.dir_B) if "png" in _file])
        self.AB_paths = list(zip(self.A_paths, self.B_paths))

        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        if opt.k_type == "test":
            self.test = True

        else:
            self.test = False
            # print(opt)
            datalen = len(self.A_paths)
            fold_size = datalen // opt.k_fold  # 每份的个数:数据总条数/折数（组数）
            val_start = opt.k_index * fold_size
            if opt.k_index != opt.k_fold - 1:
                val_end = (opt.k_index + 1) * fold_size
                self.AB_paths_valid = self.AB_paths[val_start:val_end]
                self.AB_paths_train = self.AB_paths[0:val_start] + \
                    self.AB_paths[val_end:]
            else:  # 若是最后一折交叉验证
                # 若不能整除，将多的case放在最后一折里
                self.AB_paths_valid = self.AB_paths[val_start:]
                self.AB_paths_train = self.AB_paths[0:val_start]
            if opt.k_type == "train":
                self.AB_paths = self.AB_paths_train
            elif opt.k_type == "valid":
                self.AB_paths = self.AB_paths_valid

        self.dataset_len = len(self.AB_paths)
        # crop_size should be smaller than the size of loaded image
        assert(self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / (_range)
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A_path, B_path = self.AB_paths[index]
        
        A = Image.open(os.path.join(self.dir_A, A_path)).convert('L')
        B = Image.open(os.path.join(self.dir_B, B_path)).convert('L')
        #A_array = self.normalization(np.array(A))*255.
        # print(f"A max:{np.array(A).max()}")
        #B_array = self.normalization(np.array(B))*255.
        # A = Image.fromarray(ants.n3).convert('L')
        #B = Image.fromarray(B_array).convert('L')
        random_degree = random.random()*90.-45
        translate = (0, 0)
        if not self.test:
            A = transforms.functional.affine(
                A, random_degree, translate=translate, scale=1, shear=0)  # ,fillcolor=-1.0)
            B = transforms.functional.affine(
                B, random_degree, translate=translate, scale=1, shear=0)  # ,fillcolor=-1.0)
        transform_params = get_params(self.opt, A.size)
        data_transform_A = get_transform(self.opt, transform_params, degree=random_degree, grayscale=(
            self.input_nc == 1), is_input=True)  # grayscale=False)#grayscale=(self.input_nc == 1))
        data_transform_B = get_transform(self.opt, transform_params, degree=random_degree, grayscale=(
            self.input_nc == 1), is_input=False)  # grayscale=False)#grayscale=(self.input_nc == 1))
        A = data_transform_A(A)
        B = data_transform_B(B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class AlignedDataset_4dct(BaseDataset):
    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "input")
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "output")
        self.A_paths = sorted(
            [_file for _file in os.listdir(self.dir_A) if "npy" in _file])
        self.B_paths = sorted(
            [_file for _file in os.listdir(self.dir_B) if "npy" in _file])
        self.AB_paths = list(zip(self.A_paths, self.B_paths))

        if opt.k_type == "test":
            self.test = True
        else:
            self.test = False
            datalen = len(self.A_paths)
            fold_size = datalen // opt.k_fold
            val_start = opt.k_index * fold_size
            if opt.k_index != opt.k_fold - 1:
                val_end = (opt.k_index + 1) * fold_size
                self.AB_paths_valid = self.AB_paths[val_start:val_end]
                self.AB_paths_train = self.AB_paths[0:val_start] + \
                    self.AB_paths[val_end:]
            else:
                self.AB_paths_valid = self.AB_paths[val_start:]
                self.AB_paths_train = self.AB_paths[0:val_start]
            if opt.k_type == "train":
                self.AB_paths = self.AB_paths_train
            elif opt.k_type == "valid":
                self.AB_paths = self.AB_paths_valid

        self.dataset_len = len(self.AB_paths)
        self.input_nc = 2
        self.output_nc = 9

    def __getitem__(self, index):
        A_path, B_path = self.AB_paths[index]
        A = np.load(os.path.join(self.dir_A, A_path))
        # _A = (A - A.min())/(A.max()-A.min())
        B = np.load(os.path.join(self.dir_B, B_path))
        # _B = (A - A.min())/(A.max()-A.min())
        transform_params = get_params(self.opt, A.size)
        # data_transform = [transforms.Grayscale(1), transforms.ToTensor(
        # ), transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))]
        A = data_transform_A(_A)
        B = data_transform_B(_B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.AB_paths)
