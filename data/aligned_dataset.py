import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset,get_mat
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))


        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))#grayscale=False)#grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))#grayscale=False)#grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

        """
                w, h = AB.size
        w3 = int(w / 3)
        A = AB.crop((0, 0, w3, h))
        A2 = AB.crop((w3*2, 0, w, h))

        B = AB.crop((w3, 0, w3*2, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))

        
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        A2 = A_transform(A2)

        A_final = np.array([A,A2])
        B = B_transform(B)

        return {'A': A_final, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

        """
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class AlignedDataset_mat(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.input_path = os.path.join(self.dir_AB,"inp")
        self.target_path = os.path.join(self.dir_AB,"out")

        self.input_paths = sorted(make_dataset(self.input_path, opt.max_dataset_size))  # get image paths
        self.target_paths = sorted(make_dataset(self.target_path, opt.max_dataset_size)) # get image paths

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):

        _input = self.input_paths[index]
        _target = self.target_paths[index]

        A = get_mat(_input)
        B = np.expand_dims(get_mat(_target),axis=2)

        #A_transform = transforms.Compose([transforms.ToTensor()])
        #B_transform = transforms.Compose([transforms.ToTensor()])
        A = A.type(torch.FloatTensor)
        B = B.type(torch.FloatTensor)
        return {'A': A, 'B': B, 'A_paths': _input, 'B_paths': _target}

    def __len__(self):
        return len(self.input_paths)



class AlignedDataset_npy(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.npy_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):

        _file = self.npy_paths[index]
        #A = np.expand_dims(np.load(_file)[0,0:64,0:64,0:64],axis=0)
        #B = np.load(_file)[1:10,0:64,0:64,0:64]

        A = Variable(torch.randn(2, 64, 64, 256))
        B = Variable(torch.randn(8, 64, 64, 256))
        print(f"A shape:{A.shape}")
        print(f"B shape:{B.shape}")
        #A_transform = transforms.Compose([transforms.ToTensor()])
        #B_transform = transforms.Compose([transforms.ToTensor()])
        #A = torch.from_numpy(A)
        #B = torch.from_numpy(B)
        return {'A': A, 'B': B, 'A_paths': _file, 'B_paths': _file}
    def __len__(self):
        return len(self.npy_paths)
