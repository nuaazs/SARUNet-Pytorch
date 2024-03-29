import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import SimpleITK as sitk
import matplotlib
import os

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
    
    def __init__(self,net_name,rpath="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/",epoch="best"):
        # self.organs_list = ["lens_l","lens_r","ssj_l","ssj_r","brain","brainstem","gtv","skull","soft_tissue","bone","air"]
        self.organs_list = ["soft_tissue","bone","air"]
        print(f"Generate {net_name} DataLoader")
        self.rpath = rpath
        self.SAVE_PATH = rpath+f"/{net_name}/"
        self.ROOT = self.SAVE_PATH+f"/test_{epoch}/images/"
        self._change_dir()

    def _split_ct(self,array):
        soft_tissue_array = np.zeros(array.shape)
        bone_array = np.zeros(array.shape)
        air_array = np.zeros(array.shape)
        soft_tissue_array[array>=-200] =1
        soft_tissue_array[array>400] =0
        bone_array[array>400] =1
        air_array[array<-200] = 1
        return soft_tissue_array,bone_array,air_array

    def _get_mask(self,array,label):
        output=np.zeros(array.shape)
        output[array==label] = 1
        return output

    def _get_array(self,name,pname):
        boron = [file_ for file_ in os.listdir(os.path.join(self.rpath,"seg")) if (name in file_) and (pname in file_) and ("npy" in file_)]
        #print(boron)
        boron_array = np.load(os.path.join(self.rpath,"seg",boron[0]))
        return boron_array
    
    def _get_organ_arrays(self):
        _soft_tissue,_bone,_air = self._split_ct(self.real_B_array)
        np.save(self.rpath+os.path.join(f"/seg/{self.pname}_soft_tissue.npy"),_soft_tissue)
        np.save(self.rpath+os.path.join(f"/seg/{self.pname}_bone.npy"),_bone)
        np.save(self.rpath+os.path.join(f"/seg/{self.pname}_air.npy"),_air)

    def _change_dir(self):
        os.chdir(self.SAVE_PATH)
        return 0

    def _set_seg_path(self,seg_path):
        self.SEG_PATH = seg_path
        return 0

    def _images_list(self,pname,mode):
        all_files  = os.listdir(self.ROOT)
        _list = sorted([os.path.join(self.ROOT,file_name) for file_name in all_files if (pname in file_name) and (mode in file_name)])
        return _list
    
    def _get_np_list(self,files_list):
        np_list = []
        for i in range(len(files_list)):
            hu_image = self._get_np(files_list[i])
            np_list.append(hu_image)
        return np_list

    def _get_np(self,filename):
        _np = np.array(Image.open(filename).convert("L"))/255.
        hu_np = _np*(1700+1000)-1000
        return hu_np

    def _get_error(self,real,fake):
        _error = fake - real
        _error[_error > 1000] = 1000
        _error[_error < -1000]  = -1000
        return _error
    
    def _get_mae(self):
        data= np.abs(self._get_error(self.real_B_array,self.fake_B_array))
        mae= data.mean()
        return mae

    
    def _get_label_error(self,organ):
        label_array = self._get_array(organ,self.pname)
        #print(f"label_array shape:{label_array.shape}")
        _error = self.fake_B_array[label_array>0] - self.real_B_array[label_array>0]
        
        _error[_error > 1000] = 1000
        _error[_error < -1000]  = -1000
        return _error

    def _get_label_mae(self,organ):
        data= np.abs(self._get_label_error(organ))
        mae= data.mean()
        return mae

    def _get_organs_mae(self):
        mae_dict = {}
        for organ in self.organs_list:
            mae_dict[organ]=self._get_label_mae(organ)
        return mae_dict

    def _get_output_nii(self,mode,pname):
        image_list = self._images_list(pname,mode)
        np_list = np.transpose(self._get_np_list(image_list),(1,2,0)) # 256 256 xxx
        nii = sitk.GetImageFromArray(np_list)
        return np.array(np_list),nii
    
    # def _get_output_array(self,mode,pname):
    #     image_list = self._images_list(pname,mode)
    #     np_list = np.transpose(self._get_np_list(image_list),(1,2,0))
    #     return np.array(np_list)
    

    def save_nii(self,root_path='./'):
        sitk.WriteImage(self.real_B_nii,os.path.join(root_path,self.pname+"_real_B_out.nii"))
        sitk.WriteImage(self.real_A_nii,os.path.join(root_path,self.pname+"_real_A_out.nii"))
        sitk.WriteImage(self.fake_B_nii,os.path.join(root_path,self.pname+"_fake_B_out.nii"))
        #print(f"Saved :\n\tReal A:{self.real_A_array.shape}\n\tReal B:{self.real_B_array.shape}\n\tFake B:{self.fake_B_array.shape}")
    
    def load_nii(self,pname):
        self.pname = pname
        self.real_B_array,self.real_B_nii = self._get_output_nii("real_B",pname)
        self.real_A_array,self.real_A_nii = self._get_output_nii("real_A",pname)
        self.fake_B_array,self.fake_B_nii = self._get_output_nii("fake_B",pname)
        #print(f"Load\n\tReal A:{self.real_A_array.shape}\n\tReal B:{self.real_B_array.shape}\n\tFake B:{self.fake_B_array.shape}")
        if not (os.path.exists(self.rpath+os.path.join(f"/seg/{self.pname}_soft_tissue.npy"))):
            self._get_organ_arrays()
            #print("No organ arryas. Loaded!")
        self.error = self._get_error(self.real_B_array,self.fake_B_array)
        self.mae = self._get_mae()
        self.organs_mae = self._get_organs_mae()

    def plot(self,plot_mode):
        plot_range = [0,1]
        n = 0.5
        # 第几层，可修改
        COLOR_BAR_FONT_SIZE = 6


        import matplotlib.ticker as ticker
        from matplotlib import colors

        a,b,c = self.fake_B_array.shape
        if plot_mode == "sag" or plot_mode == "cor":
            ax_aspect = b/a
        else:
            ax_aspect = 1

        func = lambda x,pos: "{:g}HU".format(x)
        fmt = ticker.FuncFormatter(func)

        plt.figure(dpi=400)
        a1 = plt.subplot(1,3,1)

        if plot_mode == "sag":
            fake_array = self.fake_B_array[:,:,int(n*c)]
            real_array = self.real_B_array[:,:,int(n*c)]
        elif plot_mode == "cor":
            fake_array = self.fake_B_array[:,int(n*b),:]
            real_array = self.real_B_array[:,int(n*b),:]
        else:
            fake_array = self.fake_B_array[int(n*a),:,:]
            real_array = self.real_B_array[int(n*a),:,:]

        im =plt.imshow(fake_array,cmap='gray')
        a1.set_aspect(ax_aspect)
        plt.axis('off')
        plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
        cb1 = plt.colorbar(im, fraction=0.03, pad=0.05, format=fmt)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb1.locator = tick_locator
        cb1.set_ticks([np.min(fake_array), -1000,0, 1000, np.max(fake_array)])
        cb1.update_ticks()

        a2 = plt.subplot(1,3,2)
        im2 =plt.imshow(real_array,cmap='gray')
        a2.set_aspect(ax_aspect)
        plt.axis('off')
        plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
        cb1 = plt.colorbar(im2, fraction=0.03, pad=0.05, format=fmt)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb1.locator = tick_locator
        cb1.set_ticks([np.min(real_array), -1000,0, 1000, np.max(real_array)])
        cb1.update_ticks()

        a3 = plt.subplot(1,3,3)
        orig_cmap = matplotlib.cm.bwr
        shifted_cmap = shiftedColorMap(orig_cmap, start=-2000, midpoint=0, stop=2000, name='shifted')
        error_array = fake_array-real_array
        
        
        divnorm3=colors.TwoSlopeNorm(vmin=np.min(error_array), vcenter=0., vmax=np.max(error_array))
        im3 =plt.imshow(error_array,cmap=orig_cmap,norm=divnorm3)
        a3.set_aspect(ax_aspect)
        plt.axis('off')
        plt.rcParams['font.size'] = COLOR_BAR_FONT_SIZE
        cb1 = plt.colorbar(im3, fraction=0.03, pad=0.05, format=fmt)
        tick_locator = ticker.MaxNLocator(nbins=3)  # colorbar上的刻度值个数
        cb1.locator = tick_locator
        cb1.set_ticks([np.min(error_array),0,np.max(error_array)])
        cb1.update_ticks()
        plt.tight_layout()
        #plt.savefig("./"+pat_num+"_"+MODE+".jpg")
        plt.show()