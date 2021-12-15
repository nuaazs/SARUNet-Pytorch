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
# @File    : datasets_maker.py
# @Describe: generate datases from raw .dcm files (mri-ct project)

import os
import re
import pydicom
import shutil
import numpy as np
from PIL import Image
import SimpleITK as sitk
import nibabel as nib


class DatasetsMaker(object):
    """Read the output (png) picture of the neural network,
       perform MAE, ME, PSNR and other calculations and drawings.

    """

    def __init__(self,net_name,root_path,parten="\d{6}"):
        """Initialization function.
           Completes the loading of data and the switching of directories.

        Args:
            net_name ([str]): [backbone name]
            rpath (str, optional): [results path]. Defaults to "/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/".
            epoch (str, optional): [Batch of interest]. Defaults to "best".
        """
        self.root_path = root_path
        self._change_dir(root_path)
        _, dirs, _ = next(os.walk(parten))
        self.dirs = [dir_name for dir_name in dirs if re.findall(parten, dir_name)!=[]]

        

    def _clasify(self):
        for patient_id,dir_name in enumerate(self.dirs,1):
            fold = os.path.join(self.root_path,dir_name)
            # MRI
            dicoms = [os.path.join(fold,dicom_name) for dicom_name in os.listdir(fold) if ".dcm" in dicom_name]
            for file_path in dicoms:
                try:
                    # 读取MRI,将不同的加权MRI分开保存
                    filename = file_path.split("\\")[-1]
                    dcm = pydicom.read_file(file_path)
                    seriesUid,seriesName = dcm.SeriesInstanceUID,dcm.SeriesDescription
                    save_path = os.path.join(fold,seriesName)
                    os.makedirs(save_path, exist_ok=True)
                    shutil.move(file_path, os.path.join(save_path,filename))
                except:
                    pass

    def _parse(self,rootdir):
        filenames = [f for f in os.listdir(rootdir) if f.endswith('.nii')]
        filenames.sort()
        filetree = {}

        for filename in filenames:
            subject, modality = filename.split('.').pop(0).split('_')[:2]

            if subject not in filetree:
                filetree[subject] = {}
                filetree[subject][modality] = filename

        return filetree


    def coregister(self,rootdir, fixed_modality, moving_modality):
        rmethod = sitk.ImageRegistrationMethod()
        rmethod.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        rmethod.SetMetricSamplingStrategy(rmethod.RANDOM)
        rmethod.SetMetricSamplingPercentage(.01)
        rmethod.SetInterpolator(sitk.sitkLinear)
        rmethod.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                numberOfIterations=200,
                                                convergenceMinimumValue=1e-6,
                                                convergenceWindowSize=10)
        rmethod.SetOptimizerScalesFromPhysicalShift()
        rmethod.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        rmethod.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        rmethod.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        filetree = self._parse(rootdir)

        for subject, modalities in filetree.items():
            print(f'{subject}:')

            if fixed_modality not in modalities or moving_modality not in modalities:
                print('-> incomplete')
                continue

            fixed_path = os.path.join(rootdir, modalities[fixed_modality])
            moving_path = os.path.join(rootdir, modalities[moving_modality])

            fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
            moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            rmethod.SetInitialTransform(initial_transform, inPlace=False)
            final_transform = rmethod.Execute(fixed_image, moving_image)
            print('-> coregistered')

            moving_image = sitk.Resample(
                moving_image, fixed_image, final_transform, sitk.sitkLinear, .0,
                moving_image.GetPixelID())
            moving_image = sitk.Cast(moving_image, sitk.sitkInt16)
            print('-> resampled')

            sitk.WriteImage(moving_image, moving_path)
            print('-> exported')

        # numpy的归一化
    def _normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / (_range)

    def _split_ct(self,array):
        """Depending on the range of CT values,
           the mask of soft tissue, bone, and air is generated.

        Args:
            array ([array]): [ct array]

        Returns:
            [soft_tissue_array]: ST
            [bone_array]: Bone
            [air_array]: Air
        """
        soft_tissue_array = np.zeros(array.shape)
        bone_array = np.zeros(array.shape)
        air_array = np.zeros(array.shape)
        soft_tissue_array[array>=-200] =1
        soft_tissue_array[array>400] =0
        bone_array[array>400] =1
        air_array[array<-200] = 1
        return soft_tissue_array,bone_array,air_array

    def _resize_image_itk(self,itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
        """
        image resize withe sitk resampleImageFilter
        :param itkimage:
        :param newSpacing:such as [1,1,1]
        :param resamplemethod:
        :return:
        """
        newSpacing = np.array(newSpacing, float)
        # originSpcaing = itkimage.GetSpacing()
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()
        factor = newSpacing / originSpcaing
        newSize = originSize / factor
        newSize = newSize.astype(np.int)
        resampler.SetReferenceImage(itkimage)  # 将输出的大小、原点、间距和方向设置为itkimage
        resampler.SetOutputSpacing(newSpacing.tolist())  # 设置输出图像间距
        resampler.SetSize(newSize.tolist())  # 设置输出图像大小
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)
        return itkimgResampled


    def DownsamplingDicomFixedResolution(self):
        src_path = r"D:/GCBRAIN/selected/niis/"
        result_path = r"D:/GCBRAIN/selected/niis/"

        heightspacing_new = 1.0
        widthspacing_new = 1.0
        image_list  = sorted([os.path.join(src_path,file) for file in os.listdir(src_path) if "web" not in file])
        for image in image_list:
            img = sitk.ReadImage(image)
            Spacing = img.GetSpacing()
            print(f"Spacing before : {Spacing}")
            thickspacing, widthspacing, heightspacing = Spacing[2], Spacing[0], Spacing[1]
            
            srcitk = self._resize_image_itk(img, newSpacing=(widthspacing_new, heightspacing_new, thickspacing),
                                    originSpcaing=(widthspacing, heightspacing, thickspacing),
                                    resamplemethod=sitk.sitkNearestNeighbor)
            
            path_ = os.path.join(result_path,image.split("/")[-1])
            
            print(f"Save path:{path_}")
            Spacing_after = srcitk.GetSpacing()
            print(f"Spacing_after = {Spacing_after}")
            sitk.WriteImage(srcitk,path_)
    
    def generate_png(self)
        # 开始写png
        i = 1

        for patient_id in range(len(target_files)):
            file_name = target_files[patient_id].split(".")[0]
            target_filepath = os.path.join(target_path,target_files[patient_id])
            target_data = np.asanyarray(nib.load(target_filepath).dataobj)

            input_filepath = os.path.join(input_path,input_files[patient_id])
            
            input_data = np.asanyarray(nib.load(input_filepath).dataobj)
            
            print(f"Input shape:{input_data.shape},Output shape:{target_data.shape}")
            
            
            
            print(f"Input Max:{input_data.max()},Output Max:{target_data.max()}")
            _a,_b,slices_num = input_data.shape
            print(target_filepath)
            print(input_filepath)
            
            
            assert input_data.shape== target_data.shape
            
            #os.makedirs("/home/zhaosheng/paper2/data/geng01_new/geng_data_01/png/", exist_ok=True)
            for slice in range(slices_num):

                target_numpy = np.array(target_data[:,:,slice])
                
                input_numpy = np.array(input_data[:,:,slice])
            
                # 删除空白图片
                if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)) :
                    continue
                    
                target_numpy[target_numpy>1700] = 1700
                target_numpy[target_numpy<-1000] = -1000
                
                target_numpy_norm = normalization(target_numpy)
                input_numpy_norm = normalization(input_numpy)
        
                if np.any(np.isnan(target_numpy_norm)) or np.any(np.isnan(input_numpy_norm)) or np.mean(target_numpy_norm)<0.01:
                    continue

                patient_id_pre = '%03d' % (patient_id+1)
                prefix='%03d' % i

                pic_array = np.hstack([input_numpy_norm*255,target_numpy_norm*255])
                img = Image.fromarray(pic_array)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((512, 256), Image.ANTIALIAS)

                img.save(r"D:/GCBRAIN/selected/images/"+file_name+"_"+prefix+".png")
                i = i+1
        print(f"Done! Total:{i} pairs")

    def _get_mask(self,array,label):
        """Get the mask with the label value.

        Args:
            array ([type]): [description]
            label ([int]): [The specified label value]

        Returns:
            [type]: [description]
        """
        output=np.zeros(array.shape)
        output[array==label] = 1
        return output

    def _get_array(self,name,pname):
        """Under the specified directory, read the specified numpy file.

        Args:
            pname ([str]): [patient num or name]
            name ([str]): [The string contained in the file name]

        Returns:
            [_array]: [numpy array which has been found.]
        """
        boron = [file_ for file_ in os.listdir(os.path.join(self.rpath,"seg")) if (name in file_) and (pname in file_) and ("npy" in file_)]
        _array = np.load(os.path.join(self.rpath,"seg",boron[0]))
        return _array

    def _get_organ_arrays(self):
        _soft_tissue,_bone,_air = self._split_ct(self.real_B_array)
        np.save(self.rpath+os.path.join(f"/seg/{self.pname}_soft_tissue.npy"),_soft_tissue)
        np.save(self.rpath+os.path.join(f"/seg/{self.pname}_bone.npy"),_bone)
        np.save(self.rpath+os.path.join(f"/seg/{self.pname}_air.npy"),_air)

    def _change_dir(self,root_path):
        os.chdir(self.root_path)
        return 0

    def _set_seg_path(self,seg_path):
        """set the seg path(path to nii the file)"""
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
        """AI is creating summary for _get_label_error

        Args:
            organ ([type]): [description]

        Returns:
            [type]: [description]
        """
        label_array = self._get_array(organ,self.pname)
        _error = self.fake_B_array[label_array>0] - self.real_B_array[label_array>0]
        _error[_error > 1000] = 1000
        _error[_error < -1000]  = -1000
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

    def _get_organs_mae(self):
        """AI is creating summary for _get_organs_mae

        Returns:
            [type]: [description]
        """
        mae_dict = {}
        for organ in self.organs_list:
            mae_dict[organ]=self._get_label_mae(organ)
        return mae_dict

    def _get_output_nii(self,mode,pname):
        """AI is creating summary for _get_output_nii

        Args:
            mode ([type]): [description]
            pname ([type]): [description]

        Returns:
            [type]: [description]
        """
        image_list = self._images_list(pname,mode)
        np_list = np.transpose(self._get_np_list(image_list),(1,2,0)) # 256 256 xxx
        nii = sitk.GetImageFromArray(np_list)
        return np.array(np_list),nii


    def save_nii(self,root_path='./'):
        """AI is creating summary for save_nii

        Args:
            root_path (str, optional): [description]. Defaults to './'.
        """
        sitk.WriteImage(self.real_B_nii,os.path.join(root_path,self.pname+"_real_B_out.nii"))
        sitk.WriteImage(self.real_A_nii,os.path.join(root_path,self.pname+"_real_A_out.nii"))
        sitk.WriteImage(self.fake_B_nii,os.path.join(root_path,self.pname+"_fake_B_out.nii"))

    def load_nii(self,pname):
        """AI is creating summary for load_nii

        Args:
            pname ([type]): [description]
        """
        self.pname = pname
        self.real_B_array,self.real_B_nii = self._get_output_nii("real_B",pname)
        self.real_A_array,self.real_A_nii = self._get_output_nii("real_A",pname)
        self.fake_B_array,self.fake_B_nii = self._get_output_nii("fake_B",pname)
        if not (os.path.exists(self.rpath+os.path.join(f"/seg/{self.pname}_soft_tissue.npy"))):
            self._get_organ_arrays()
            #print("No organ arryas. Loaded!")
        self.error = self._get_error(self.real_B_array,self.fake_B_array)
        self.mae = self._get_mae()
        self.organs_mae = self._get_organs_mae()

    def plot(self,plot_mode):
        """AI is creating summary for plot

        Args:
            plot_mode ([type]): [description]
        """
        plot_range = [0,1]
        n = 0.5
        # 第几层，可修改
        COLOR_BAR_FONT_SIZE = 6

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
