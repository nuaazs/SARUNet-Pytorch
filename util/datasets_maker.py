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

    def __init__(self,net_name,root_path,parten="\d{6}"):
        self.root_path = root_path
        self._change_dir(root_path)
        _, dirs, _ = next(os.walk(parten))
        self.dirs = [dir_name for dir_name in dirs if re.findall(parten, dir_name)!=[]]

    def _clasify(self):
        """将不同模态的dcm文件分开保存至 root_path下的不同的子文件夹。
        """
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
        return 0

    def _generate_nii(self):
        """将不同模态的dcm文件转换为nii文件保存至root_path。
        """
        for patient_id,dir_name in enumerate(self.dirs,1):
            fold = os.path.join(self.root_path,dir_name)
            _,dirs, _ = next(os.walk(fold))
            dirs = [dir_name for dir_name in dirs]
            patient_id = '%03d' % patient_id
            print(f"{patient_id} -> {patient_id}")

            for mri_dir in [_dirs for _dirs in dirs if "Reg" not in _dirs and "Contour" not in _dirs and "Set" not in _dirs and "Doses" not in _dirs]:
                try:
                    mri_path = os.path.join(fold,mri_dir)
                    mri_reader = sitk.ImageSeriesReader()
                    mri_dicoms = mri_reader.GetGDCMSeriesFileNames(mri_path)
                    mri_reader.SetFileNames(mri_dicoms)
                    mri_img = mri_reader.Execute()
                    mri_size = mri_img.GetSize()
                    mri_img = sitk.Cast(mri_img, sitk.sitkFloat32)
                    if "HeadSeq" in mri_dir:
                        mode = "ct"
                    else:
                        mode = mri_dir.split("\\")[-1]
                    sitk.WriteImage(mri_img, self.root_path+"_"+mode+".nii")
                except:
                    pass
        return 0

    def _parse(self,rootdir):
        """get filetree for coregister

        Args:
            rootdir ([str]): [description]

        Returns:
            [type]: [filetree]
        """
        filenames = [f for f in os.listdir(rootdir) if f.endswith('.nii')]
        filenames.sort()
        filetree = {}
        for filename in filenames:
            subject, modality = filename.split('.').pop(0).split('_')[:2]

            if subject not in filetree:
                filetree[subject] = {}
                filetree[subject][modality] = filename
        return filetree

    def _coregister(self, fixed_modality="t1", moving_modality="ct"):
        """Registration, overwriting the original nii file

        Args:
            fixed_modality (str, optional): [fixed modality]. Defaults to "t1".
            moving_modality (str, optional): [moving modality]. Defaults to "ct".
        """
        rootdir = self.root_path
        self.target_files = sorted([target for target in os.listdir(rootdir) if moving_modality in target])
        self.input_files = sorted([input_ for input_ in os.listdir(rootdir) if fixed_modality in input_])
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
        return 0

    def _normalization(self,data):
        """normalize numpy array.

        Args:
            data ([numpy]): [numpy]

        Returns:
            [numpy]: [noralized numpy]
        """
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / (_range)

    def _resize_image_itk(self,itkimage, newSpacing, originSpcaing, resamplemethod=sitk.sitkNearestNeighbor):
        """resample itk image.Make sure the dataset has the same voxel size.

        Args:
            itkimage ([image]): [itkimage]
            newSpacing ([tuple]): [newSpacing]
            originSpcaing ([tuple]): [originSpcaing]
            resamplemethod ([method], optional): [sitk resamplemethod]. Defaults to sitk.sitkNearestNeighbor.

        Returns:
            [image]: [resampled image]
        """
        newSpacing = np.array(newSpacing, float)
        # originSpcaing = itkimage.GetSpacing()
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()
        factor = newSpacing / originSpcaing
        newSize = originSize / factor
        newSize = newSize.astype(np.int)
        resampler.SetReferenceImage(itkimage)  # Set the size, origin, spacing, and orientation of the output to itkimage
        resampler.SetOutputSpacing(newSpacing.tolist())  # Set the output image spacing
        resampler.SetSize(newSize.tolist())  # Set the size of the output image
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)
        return itkimgResampled

    def _DownsamplingDicomFixedResolution(self,heightspacing_new=1.0,widthspacing_new=1.0):
        """Downsampling Dicom Fixed Resolution.Make sure the dataset has the same voxel size.

        Args:
            heightspacing_new (float, optional): [description]. Defaults to 1.0.
            widthspacing_new (float, optional): [description]. Defaults to 1.0.
        """
        src_path=self.root_path
        result_path=self.root_path
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
            # print(f"Save path:{path_}")
            Spacing_after = srcitk.GetSpacing()
            print(f"Spacing_after = {Spacing_after}")
            sitk.WriteImage(srcitk,path_)
        return 0

    def generate_png(self,output_path):
        """generate pngs in generate_png.For Dataloader of Pytorch.

        Args:
            output_path ([str]): [output path]
        """
        os.makedirs(output_path, exist_ok=True)
        i = 1
        for patient_id in range(len(self.target_files)):
            file_name = self.target_files[patient_id].split(".")[0]
            target_filepath = os.path.join(self.target_path,self.target_files[patient_id])
            target_data = np.asanyarray(nib.load(target_filepath).dataobj)
            input_filepath = os.path.join(self.input_path,self.input_files[patient_id])
            input_data = np.asanyarray(nib.load(input_filepath).dataobj)
            # print(f"Input shape:{input_data.shape},Output shape:{target_data.shape}")
            # print(f"Input Max:{input_data.max()},Output Max:{target_data.max()}")
            _a,_b,slices_num = input_data.shape
            assert input_data.shape== target_data.shape
            for slice in range(slices_num):
                target_numpy = np.array(target_data[:,:,slice])
                input_numpy = np.array(input_data[:,:,slice])
                # Delete blank pictures
                if (np.max(target_numpy) == np.min(target_numpy)) or (np.max(input_numpy) == np.min(input_numpy)) :
                    continue
                target_numpy[target_numpy>1700] = 1700
                target_numpy[target_numpy<-1000] = -1000
                target_numpy_norm = self._normalization(target_numpy)
                input_numpy_norm = self._normalization(input_numpy)
                if np.any(np.isnan(target_numpy_norm)) or np.any(np.isnan(input_numpy_norm)) or np.mean(target_numpy_norm)<0.01:
                    continue
                patient_id_pre = '%03d' % (patient_id+1)
                prefix='%03d' % i
                pic_array = np.hstack([input_numpy_norm*255,target_numpy_norm*255])
                img = Image.fromarray(pic_array)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((512, 256), Image.ANTIALIAS)
                img.save(output_path + r"/"+file_name+"_"+prefix+".png")
                i = i+1
        print(f"Done! Total:{i} pairs")
        return 0

    def _change_dir(self,root_path):
        """Change dir

        Args:
            root_path ([str]): [Destination path]

        """
        os.chdir(self.root_path)
        return 0

    def automake(self,png_path):
        self._clasify()
        self._generate_nii()
        print("  -> Please perform manual screening.")
        input_str = input("Okay?(Y(es)/N(o))")
        if "y" in input_str.lower():
            self._DownsamplingDicomFixedResolution()
            self._coregister()
            self.generate_png(png_path)
        else:
            print("Exit!")
