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
# @File    : DVH.py
# @Describe: Get DVH from the dose results of MC(mcnp/topas/G4).

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nb
import numpy as np

class DVH(object):
    """Get DVH from the dose results of MC."""


    def __init__(self,MC_out_file_path,organs_list,organs_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/seg/",pname="001",filetype="csv"):
        """ Init fun. Get `organs_array_dict` and xxx_dose array.

        Args:
            MC_out_file_path ([str]): [The path to the output file]
            organs_list ([str list]): [organ's name list]
        """

        self.organs = organs_list
        self.pname = pname
        self.organs_array_dict = {}
        self.organs_dvh_dict = {}
        self.organs_path = organs_path
        for organ in organs_list:
            print(f"Now {organ}.")
            self.organs_array_dict[organ] = self._get_organ_array(organ,self.pname)
        if filetype=="csv":
            self.total_dose_real,self.total_dose_fake = self._get_dose_array(MC_out_file_path)
        elif filetype == "mat":
            self.total_dose = self._get_dose_array_from_mat(MC_out_file_path) # , self.boron_dose, self.fast_n_dose, self.N_dose
        else:
            print("Error filetype.")


    def _array_from_MC(self,MC_OUT_FILE,mode="txt"):
        """Get a two-dimensional numpy array from the output file
           of Monte Carlo software.

        Args:
            MC_OUT_FILE ([str]): [The path to the output file]
        Returns:
            [array]: [The result of this output file in a two-dimensional array format]
        """
        if mode == "txt":
            if os.path.isfile(MC_OUT_FILE[:-4]+".npy"):
                filename_ = MC_OUT_FILE[:-4]+".npy"
                print(f"\t->Loading npy file : {filename_}")
                return np.load(filename_)

            print(f"\t->Reading MC output file : {MC_OUT_FILE}")
            with open(MC_OUT_FILE,encoding = 'utf-8') as f:
                data = np.loadtxt(f,delimiter = ",", skiprows = 8)
            output = data[:,3].reshape(256,256,187).transpose(2,0,1)
            np.save(MC_OUT_FILE[:-4],output)
            print(f"\t->Saving MC output file : {MC_OUT_FILE}")
            return output

        if mode == "mat":
            data = scio.loadmat(MC_OUT_FILE)
            return data[[ky for ky in data.keys() if "_" not in ky][0]].transpose(2,0,1)


    def _get_organ_array(self,organ,pname):
        """Get a mask for the specified organ.

        Args:
            filepath ([str]): [organ arrays saved path]
            organ ([str]): [The name of the organ]
        Returns:
            [array]: [mask for the specified organ]
        """
        return self._get_array(self.organs_path,pname+"_"+organ)



    def _get_dose_array(self,outpath):
        """Get total dose array, and Boron/fast n/gamma/nitrogen dose.

        Args:
            outpath ([str]): [ ([str]): [The path to the MC output file]]

        Returns:
            [array]: [ Total dose array for BNCT]
        """
        r=36
        n=6.5
        xx = n*r*3.1415926*30*60
        t=60 #tumour
        s=25 #skin
        nt=18#NT Boron concentration
        self.boron_array_real = self._array_from_MC(os.path.join(outpath,"boron_real.csv"))
        self.fast_array_real = self._array_from_MC(os.path.join(outpath,"fast_real.csv"))
        self.gamma_array_real = self._array_from_MC(os.path.join(outpath,"gamma_real.csv"))
        self.nitrogen_array_real = self._array_from_MC(os.path.join(outpath,"nitrogen_real.csv"))
        self.boron_array_fake = self._array_from_MC(os.path.join(outpath,"boron_fake.csv"))
        self.fast_array_fake = self._array_from_MC(os.path.join(outpath,"fast_fake.csv"))
        self.gamma_array_fake = self._array_from_MC(os.path.join(outpath,"gamma_fake.csv"))
        self.nitrogen_array_fake = self._array_from_MC(os.path.join(outpath,"nitrogen_fake.csv"))
        img = nb.load("/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/001/output_nii/001_real.nii")
        ct_array = np.asanyarray(img.dataobj)
        self.ct = np.array(ct_array)
        #self.ct = self._get_organ_array("ct")
        
        self.gtv = self._get_organ_array("gtv",self.pname)
#         self.ssj_l = self._get_organ_array("ssj_l",self.pname)
#         self.ssj_r = self._get_organ_array("ssj_r",self.pname)
        self.lens_l = self._get_organ_array("lens_l",self.pname)
        self.lens_r = self._get_organ_array("lens_r",self.pname)
        
        boron_concentration = np.zeros(self.ct.shape)+18*1.4
        print(boron_concentration.shape)
        boron_concentration[self.gtv==1]=60*3.8
        #boron_concentration[self.skin==1]=25*2.5
        boron_concentration[self.ct<-999]=0 # 空气
        boron_concentration[self.lens_r==1]=10*1.4
        boron_concentration[self.lens_l==1]=10*1.4
        
        
        boron_dose_real = self.boron_array_real*boron_concentration*xx
        gamma_dose_real = self.gamma_array_real*xx
        proton_dose_real = (self.fast_array_real+self.nitrogen_array_real)*3.2*xx
        total_dose_real=boron_dose_real+gamma_dose_real+proton_dose_real

        boron_dose_fake = self.boron_array_fake*boron_concentration*xx
        gamma_dose_fake = self.gamma_array_fake*xx
        proton_dose_fake = (self.fast_array_fake+self.nitrogen_array_fake)*3.2*xx
        total_dose_fake=boron_dose_fake+gamma_dose_fake+proton_dose_fake

        return total_dose_real,total_dose_fake

    def _get_dose_array_from_mat(self,outpath):
        """Get total dose array, and Boron/fast n/gamma/nitrogen dose.

        Args:
            outpath ([str]): [ ([str]): [The path to the MC output file]]

        Returns:
            [array]: [ Total dose array for BNCT]
        """
        r=36
        n=6.5
        xx = n*r*3.1415926*30*60
        t=50 #肿瘤
        s=25 #皮肤
        nt=18#NT硼浓度
        self.boron_array = self._array_from_MC(os.path.join(outpath,"boron.mat"),mode="mat")
        self.proton_array = self._array_from_MC(os.path.join(outpath,"proton.mat"),mode="mat")
        self.gamma_array = self._array_from_MC(os.path.join(outpath,"gamma.mat"),mode="mat")

        print(self.boron_array.shape)
        print(self.proton_array.shape)
        print(self.boron_array.shape)
        img = nb.load("/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/001/output_nii/001_real.nii")
        ct_array = np.asanyarray(img.dataobj)
        ct = np.array(ct_array)

        print(ct.shape)
        #self.ct = self._get_organ_array("ct")
        self.ct = ct
        self.gtv = self._get_organ_array("gtv",self.pname)
        total_dose= np.zeros(self.ct.shape)
        total_dose[self.ct<-999] = self.boron_array[self.ct<-999]*xx* \
                        + self.proton_array[self.ct<-999]*xx*3.2 \
                        + self.gamma_array[self.ct<-999]*xx
        total_dose = self.boron_array*xx*nt \
                    + self.proton_array*xx*3.2 \
                    + self.gamma_array*xx

        total_dose[self.gtv==1] = self.boron_array[self.gtv==1]*xx*t \
                    + self.proton_array[self.gtv==1]*xx*3.2 \
                    + self.gamma_array[self.gtv==1]*xx
        return total_dose

    def _get_dvh_data(self,dose):
        """Get dvh information for all organs and save it in a dictionary.
        """
        for organ_name in self.organs:
            print(organ_name)
            xy_list = self._get_organ_dvh(self.organs_array_dict[organ_name],dose)
            xy_list.append(organ_name) ## [xx,yy,label]
            self.organs_dvh_dict[organ_name] = xy_list
        return 0

    def _get_organ_dvh(self,organ_array,dose):
        """Obtain the curve x, y coordinate value (1,000 points)
            of the DVH of the organ.

        Args:
            organ_array ([array]): [organ array(mask)]]

        Returns:
            [list]: [coordinate value (1,000 points) ]
        """
        if len(organ_array[organ_array>0]) == 0:
            xx = np.linspace(0,1,1000)
            yy = np.linspace(0,1,1000)
            xx= xx.tolist()
            yy= yy.tolist()
            return xx,yy
        out_dose = np.zeros(organ_array.shape)
        out_dose[organ_array>0] = dose[organ_array>0]
        max_dose = dose.max()
        xx = np.linspace(0,max_dose,1001)
        xx = xx[:1000]
        yy = []
        for x in xx:
            yy.append(len(out_dose[out_dose>x]) / len(organ_array[organ_array>0]))
        xx= xx.tolist()
        yy[-1]=0
        return [xx,yy]


    def plot_dvh(self):
        """Visualize the dvh diagram of the organs
        """
        
        plt.figure(dpi=150)
        
        self._get_dvh_data(self.total_dose_real)
        for organ_name in self.organs_array_dict.keys():
            _dvh_info = self.organs_dvh_dict[organ_name]
            plt.plot(_dvh_info[0],_dvh_info[1],label=_dvh_info[2],linewidth=1)

        self._get_dvh_data(self.total_dose_fake)
        for organ_name in self.organs_array_dict.keys():
            _dvh_info = self.organs_dvh_dict[organ_name]
            plt.plot(_dvh_info[0],_dvh_info[1],label=_dvh_info[2],linewidth=1)

        plt.legend()
        plt.xlabel("Dose(Gy)")
        plt.ylabel("Ratio of total structure volume(%)")
        plt.title("DVH")
        plt.savefig("./dvh.png")
        plt.show()
        return 0


    def _get_array(self,fold,name):
        """Under the specified directory, read the specified numpy file.

        Args:
            fold ([str]): [The path to look for]
            name ([str]): [The string contained in the file name]

        Returns:
            [array]: [numpy array which has been found.]
        """
        _filename = [file_ for file_ in os.listdir(fold) if name in file_]
        _array = np.load(os.path.join(fold,_filename[0]))
        return _array

if __name__ == "__main__":
    mcnp_output_path = "/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/dose/001/"
    dvh = DVH(mcnp_output_path,["lens_l","lens_r","gtv","brain","skull"]) #
    print(f"Total Dose Shape:{dvh.total_dose_real.shape}")
#     plt.figure()
#     plt.imshow(dvh.total_dose[100])
#     plt.title("Total Dose")
#     plt.axis('off')
#     plt.colorbar()
#     plt.show()

#     plt.figure()
#     plt.imshow(dvh.boron_dose)
#     plt.title("Boron Dose")
#     plt.axis('off')
#     plt.colorbar()
#     plt.show()

    dvh.plot_dvh()
