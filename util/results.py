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
# @File    : util/result_analyzer.py
# @Describe: Analysis of neural network results (mri-ct project)

import os
from PIL import Image
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from glob import glob
from os import listdir
from os.path import splitext
import nibabel as nb
import nibabel as nib
import numpy as np
from nibabel.viewers import OrthoSlicer3D
import scipy.io as io
import ants

    
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

class DataLoader(object):
    """Read the output (png) picture of the neural network, 
       perform MAE, ME, PSNR and other calculations and drawings.

    """

    def __init__(self,
                 net_name,
                 rpath="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/",
                 seg_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/seg/",
                 raw_img_path = "/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/real_ct_niis_0327/",
                 nii_save_path = "/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/fake_ct_niis/",
                 test_list = ['003', '099', '033', '136', '011', '086', '077', '102', '016', '022']):
        
        """Initialization function.
           Completes the loading of data and the switching of directories.

        Args:
            net_name ([str]): [backbone name]
            rpath (str, optional): [results path]. Defaults to "/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/".

        """
        self.organs_list = ["len_l","len_r","brain","brainstem","skull","on_l","on_r","eye_l","eye_r","head","air","all","skin"]
        self.rpath = rpath
        self.net_name = net_name
        self.SAVE_PATH = os.path.join(rpath,net_name)
        self.test_list = test_list
        self.ROOT = os.path.join(self.SAVE_PATH,"temp_pics")
        self.SEG_PATH =seg_path
        self.raw_img_path = raw_img_path
        self.nii_save_path = nii_save_path
        
        self.files_all = []
        for index in range(5):
            path = os.path.join(self.ROOT,f"k{index}_300")
            files = sorted([_file for _file in os.listdir(path)],key=self._get_slice_num)
            for xx in files:
                if index>0 and (xx.split("/")[-1].split("_")[0] in self.test_list):
                    continue
                # else:
                #     print(f"{xx.split("/")[-1].split("_")[0]}")
                if (xx not in self.files_all):
                    self.files_all.append(os.path.join(path,xx))

        
    def _get_array(self,name,pname):
        """Under the specified directory, read the specified numpy file.

        Args:
            pname ([str]): [patient num or name]
            name ([str]): [The string contained in the file name]

        Returns:
            [_array]: [numpy array which has been found.]
        """
        if name == "all":
            head = self._get_array("head",pname)
            return np.ones(head.shape)
        if name == "air":
            head = self._get_array("head",pname)
            return head*(-1)+1
        boron = [file_ for file_ in os.listdir(os.path.join(self.rpath,"seg")) if (name in file_) and (pname in file_) and ("npy" in file_)]
        #print(f"\t{name}")
        _array = np.load(os.path.join(self.rpath,"seg",boron[0]))
        return _array
    


    def _normalization(self,data_inp):
        """normalize numpy array.

        Args:
            data ([numpy]): [numpy]

        Returns:
            [numpy]: [noralized numpy]
        """
        data = data_inp.copy()
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / (_range)

    def _get_slice_num(self,filepath):
        return int(filepath.split("/")[-1].split("/")[-1].split("_")[1])
    
    def _get_brain_mean(self,_array,pname):
        brain_mask = self._get_array("brain",pname)
        return _array[brain_mask == 1].mean()
    def _get_brain_mask(self,pname):
        brain_mask = self._get_array("brain",pname)
        return brain_mask
    

    
    def _get_ants_img(self,pname):
        
        all_files  = self.files_all
        
        real_list = sorted([file_path for file_path in all_files if (file_path.split("/")[-1].startswith(pname) ) and ("real_B" in file_path)],key=self._get_slice_num)
        #print(real_list)
        fake_list = sorted([file_path for file_path in all_files if (file_path.split("/")[-1].startswith(pname) ) and ("fake_B" in file_path)],key=self._get_slice_num)
        #print(fake_list)
        raw_img = ants.image_read(os.path.join(self.raw_img_path,f"{pname}.nii"))
        real_np_list,fake_np_list = [],[]
        brain_mask = self._get_brain_mask(pname)
        for index in range(len(real_list)):
            # brain = brain_mask[:,:,index]
            
            real_img = real_list[index]
            fake_img = fake_list[index]
            real_np = np.array(Image.open(real_img).convert("L"))/255.*1800-1000
            fake_np = np.array(Image.open(fake_img).convert("L"))/255.*1800-1000

            real_np_list.append(real_np)
            fake_np_list.append(fake_np)
        real_np_list = np.array(real_np_list).transpose(1,2,0)
        fake_np_list = np.array(fake_np_list).transpose(1,2,0)
        
        real_np_list[real_np_list>1700] = 1700
        real_np_list[real_np_list<-1000] = -1000
        
        fake_np_list[fake_np_list>1700] = 1700
        fake_np_list[fake_np_list<-1000] = -1000
        a_real = ants.from_numpy(real_np_list)
        a_real.set_spacing(raw_img.spacing)
        
        a_fake = ants.from_numpy(fake_np_list)
        a_fake.set_spacing(raw_img.spacing)
        return a_real,a_fake
    
    def _get_np_list(self,files_list):
        np_list = []
        for i in range(len(files_list)):
            hu_image = self._get_np(files_list[i])
            np_list.append(hu_image)
        return np.array(np_list)

    def _get_np(self,filename):
        _np = np.array(Image.open(filename).convert("L"))/255.
        _np = self._normalization(_np)
        _np[_np>1] = 1
        _np[_np<0] = 0
        hu_np = _np*(1700+1000)-1000
        return hu_np

    def _get_error(self,real,fake):
        _error = fake - real
        return _error
    
    def _get_rmse(self):
        return np.sqrt(((self.fake_B_array-self.real_B_array) ** 2).mean())
    
    def _get_mae(self):
        data= np.abs(self._get_error(self.real_B_array,self.fake_B_array))
        mae= data.mean()
        return mae

    def _get_me(self):
        data= self._get_error(self.real_B_array,self.fake_B_array)
        me= data.mean()
        return me
    
    def _get_label_error(self,organ):
        """AI is creating summary for _get_label_error

        Args:
            organ ([type]): [description]

        Returns:
            [type]: [description]
        """
        label_array = self._get_array(organ,self.pname)
        _error = self.fake_B_array[label_array>0] - self.real_B_array[label_array>0]
        return _error

    def _get_label_mae(self,organ):
        """AI is creating summary for _get_label_mae

        Args:
            organ ([type]): [description]

        Returns:
            [type]: [description]
        """
        data= np.abs(self._get_label_error(organ))
        mae= data.mean()
        return mae

    def _get_label_me(self,organ):
        """AI is creating summary for _get_label_mae

        Args:
            organ ([type]): [description]

        Returns:
            [type]: [description]
        """
        data= self._get_label_error(organ)
        me= data.mean()
        return me
    
    def _get_label_rmse(self,organ):
        """AI is creating summary for _get_label_mae

        Args:
            organ ([type]): [description]

        Returns:
            [type]: [description]
        """
        data= self._get_label_error(organ)
        rmse= np.sqrt((data** 2).mean())
        return rmse
    
    def _get_organs_mae(self):
        """AI is creating summary for _get_organs_mae

        Returns:
            [type]: [description]
        """
        mae_dict = {}
        for organ in self.organs_list:
            mae_dict[organ]=self._get_label_mae(organ)
        return mae_dict
    
    def _get_organs_me(self):
        """AI is creating summary for _get_organs_mae

        Returns:
            [type]: [description]
        """
        me_dict = {}
        for organ in self.organs_list:
            me_dict[organ]=self._get_label_me(organ)
        return me_dict
    
    def _get_organs_rmse(self):
        """AI is creating summary for _get_organs_mae

        Returns:
            [type]: [description]
        """
        rmse_dict = {}
        for organ in self.organs_list:
            rmse_dict[organ]=self._get_label_rmse(organ)
        return rmse_dict
    
    def _get_output_img(self,pname):
        """AI is creating summary for _get_output_img

        Args:
            mode ([type]): [description]
            pname ([type]): [description]

        Returns:
            [type]: [description]
        """
        img_real,img_fake = self._get_ants_img(pname)
        array_real = img_real.numpy()
        array_fake = img_fake.numpy()
        return array_real,array_fake,img_real,img_fake
    def _save_output_nii(self,pname,real_img,fake_img):
        fake_img[fake_img<-1000] = -1000
        fake_img[fake_img>1700] = 1700
        os.makedirs(self.nii_save_path,exist_ok=True)
        ants.image_write(fake_img,os.path.join(self.nii_save_path,f"{self.net_name}_{pname}.nii"))
        return 0
        
        
        
    
    def load_nii(self,pname):
        """AI is creating summary for load_nii

        Args:
            pname ([type]): [description]
        """
        self.pname = pname
        self.real_B_array,self.fake_B_array,self.real_B_img,self.fake_B_img = self._get_output_img(pname)
        
        self.error = self._get_error(self.real_B_array,self.fake_B_array)
        self.mae = self._get_mae()
        self.rmse = self._get_rmse()
        self.me = self._get_me()
        self.organs_mae = self._get_organs_mae()
        self.organs_me = self._get_organs_me()
        self.organs_rmse = self._get_organs_rmse()
        self._save_output_nii(pname,self.real_B_img,self.fake_B_img)