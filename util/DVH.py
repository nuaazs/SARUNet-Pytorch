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

import matplotlib.pyplot as plt
import scipy.io as scio
import nibabel as nb
import numpy as np
import os

class DVH(object):
    """Get DVH from the dose results of MC."""
    def __init__(self,MC_out_file_path,
                 organs_list=["lens_l","lens_r","gtv","brain","skull","ssj_l","ssj_r","skin"],
                 organs_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/seg/",
                 pname="001",
                 filetype="csv",
                 #shape=(256,256,180),
                 redundancy=True,
                 root_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/"
                 ):
        """ Init fun. Get `organs_array_dict` and xxx_dose array.

        Args:
            MC_out_file_path ([str]): [The path to the output file]
            organs_list ([str list]): [organ's name list]
        """
        self.organs = organs_list
        self.pname = pname
        self.nowmode = "" #
        self.organs_array_dict = {}
        self.organs_dvh_dict_real = {}
        self.organs_dvh_dict_fake = {}
        # self.shape=shape
        self.redundancy = redundancy
        self.organs_path = organs_path
        self.root_path = root_path
        self.D_info = {}
        img = nb.load(os.path.join(self.root_path,f"{pname}/output_nii/{pname}_real.nii"))
        ct_array = np.asanyarray(img.dataobj)
        self.ct = np.array(ct_array)

        for organ in organs_list:
            print(f"Now {organ}.")
            self.organs_array_dict[organ] = self._get_organ_array(organ,self.pname)
        if filetype=="csv":
            self.total_dose_real,self.total_dose_fake = self._get_dose_array_from_csv(MC_out_file_path)
        elif filetype == "mat":
            self.total_dose_real,self.total_dose_fake = self._get_dose_array_from_mat(MC_out_file_path)
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
                array_ = np.load(filename_)
                print(f"\t->Shape:{array_.shape}")
                return array_

            print(f"\t->Reading MC output file : {MC_OUT_FILE}")
            with open(MC_OUT_FILE,encoding = 'utf-8') as f:
                data = np.loadtxt(f,delimiter = ",", skiprows = 8)
            output = data[:,3].reshape(256,256,-1).transpose(2,0,1)
            np.save(MC_OUT_FILE[:-4],output)
            print(f"\t->Saving MC output file : {MC_OUT_FILE}")
            print(f"\t->Shape:{output.shape}")
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

        if organ == "skin":
            organ = "skin_"+self.nowmode
        return self._get_array(self.organs_path,pname+"_"+organ)

    def _get_dose_array_from_csv(self,outpath):
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

        #self.ct = self._get_organ_array("ct")

        self.gtv = self._get_organ_array("gtv",self.pname)

        # real
        self.nowmode = "real"
        self.skin_real = self._get_organ_array("skin",self.pname)


        # fake
        self.nowmode = "fake"
        self.skin_fake = self._get_organ_array("skin",self.pname)


        self.lens_l = self._get_organ_array("lens_l",self.pname)
        self.lens_r = self._get_organ_array("lens_r",self.pname)


        boron_concentration = np.zeros(self.ct.shape)+18*1.4
        if self.redundancy:
            print(boron_concentration.shape)
        boron_concentration[self.gtv==1]=60*3.8

        boron_concentration[self.ct<-999]=0 # 空气
        boron_concentration[self.lens_r==1]=10*1.4
        boron_concentration[self.lens_l==1]=10*1.4

        # real
        boron_concentration[self.skin_fake==1]=18*1.4
        boron_concentration[self.skin_real==1]=25*2.5
        boron_concentration[self.gtv==1]=60*3.8
        boron_concentration[self.lens_r==1]=10*1.4
        boron_concentration[self.lens_l==1]=10*1.4
        boron_concentration[self.ct<-999]=0 # 空气
        boron_dose_real = self.boron_array_real*boron_concentration*xx
        gamma_dose_real = self.gamma_array_real*xx
        proton_dose_real = (self.fast_array_real+self.nitrogen_array_real)*3.2*xx
        total_dose_real=boron_dose_real+gamma_dose_real+proton_dose_real

        # fake
        boron_concentration[self.skin_real==1]=18*1.4
        boron_concentration[self.skin_fake==1]=25*2.5
        boron_concentration[self.gtv==1]=60*3.8
        boron_concentration[self.lens_r==1]=10*1.4
        boron_concentration[self.lens_l==1]=10*1.4
        boron_concentration[self.ct<-999]=0 # 空气
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
        if self.redundancy:
            print(self.boron_array.shape)
            print(self.proton_array.shape)
            print(self.boron_array.shape)

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

    def _get_dvh_data(self,dose,mode="real"):
        """Get dvh information for all organs and save it in a dictionary.
        """

        for organ_name in self.organs:
            if self.redundancy:
                print(organ_name)
            xy_list,D_info = self._get_organ_dvh(self.organs_array_dict[organ_name],dose)
            if self.redundancy:
                print(f"Mode:{mode} and D_info:\n")
                print(D_info)
            self.D_info[organ_name+"_"+mode] = D_info
            xy_list.append(organ_name) ## [xx,yy,label]
            if mode == "real":
                self.nowmode = "real"
                self.organs_dvh_dict_real[organ_name] = xy_list
            elif mode == "fake":
                self.nowmode = "fake"
                self.organs_dvh_dict_fake[organ_name] = xy_list
            else:
                print("!! Error mode in _get_dvh_data.")
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
        out_dose = out_dose[organ_array>0]
        # print(f"Length os out_dose:{out_dose.shape}")
        
        max_dose = np.max(out_dose)
        mean_dose = np.mean(out_dose)
        xx = np.linspace(0,np.max(dose),1001)
        xx = xx[:1000]
        yy = []
        Dmax = max_dose
        Dmean = mean_dose
        D100,D98,D95,D50,D2,D0 = 0,0,0,0,0,0
        
        notfindD95,notfindD100,notfindD0,notfindD50,notfindD2,notfindD98 = True,True,True,True,True,True
        for x in xx:
            y_value = len(out_dose[out_dose>x]) / len(organ_array[organ_array>0])
            if y_value<0.95 and notfindD95:
                D95 = x
                notfindD95 = False
            if y_value<0.99 and notfindD100:
                D100 = x
                notfindD100 = False
            if y_value<0.01 and notfindD0:
                D0 = x
                notfindD0 = False
            if y_value<0.50 and notfindD50:
                D50 = x
                notfindD0 = False
            if y_value<0.02 and notfindD2:
                D2 = x
                notfindD0 = False
            if y_value<0.98 and notfindD98:
                D98 = x
                notfindD0 = False
            yy.append(y_value)


        xx= xx.tolist()
        yy[-1]=0
        D_info = {"D95":D95,"D100":D100,"D0":D0,"D50":D50,"D2":D2,"D98":D98,"Dmax":Dmax,"Dmean":Dmean}
        return [xx,yy],D_info

    def plot_dvh(self,reload=False,show_pic=False):
        colors = [
          "green",
          "sienna",
          "firebrick",
          "darkmagenta",

          "darkblue",
          "peru",
          "teal",
          "slategrey",
          "darkkhaki",
          "C",
          "gold",
          "olive"]

        organ_labels  = {"lens_l":"Lens(L)",
                 "lens_r":"Lens(L)",
                 "ssj_l":"ON(L)",
                 "ssj_r":"ON(R)",
                 "brain":"Brain",
                 "brainstem":"BrainStem",
                 "gtv":"GTV",
                 "skull":"Skull",
                 "soft_tissue":"Soft tissue",
                 "bone":"Bone",
                 "air":"Air",
                 "skin":"Skin"}

        plt.style.use('ggplot')
        fg=plt.figure(dpi=400)
        ax=fg.add_subplot(1,1,1)
        fake_cache = f"/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/dose/{self.pname}/fake_dvh.npy"
        real_cache = f"/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/dose/{self.pname}/real_dvh.npy"
        dvh_cache = f"/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/dose/{self.pname}/dvh.npy"
#         if reload or (not (os.path.isfile(real_cache))):
#             self._get_dvh_data(self.total_dose_real,mode="real")
#             np.save(real_cache, self.organs_dvh_dict_real)
#         if reload or (not (os.path.isfile(fake_cache))):
#             self._get_dvh_data(self.total_dose_fake,mode="fake")
#             np.save(fake_cache, self.organs_dvh_dict_fake)
        if reload or (not (os.path.isfile(dvh_cache))):
            self._get_dvh_data(self.total_dose_real,mode="real")
            self._get_dvh_data(self.total_dose_fake,mode="fake")
            np.save(real_cache, self.organs_dvh_dict_real)
            np.save(fake_cache, self.organs_dvh_dict_fake)
            np.save(dvh_cache, self.D_info)

        self.organs_dvh_dict_real = np.load(real_cache,allow_pickle=True).item()
        self.organs_dvh_dict_fake = np.load(fake_cache,allow_pickle=True).item()
        self.D_info = np.load(dvh_cache,allow_pickle=True).item()
        
        if show_pic:
            for idx,organ_name in enumerate(self.organs_array_dict.keys()):
                _dvh_info_real = self.organs_dvh_dict_real[organ_name]
                ax.plot(_dvh_info_real[0],_dvh_info_real[1],linewidth=2,color=colors[idx],linestyle='-',label=organ_labels[organ_name]+" Real")


                _dvh_info_fake = self.organs_dvh_dict_fake[organ_name]
                ax.plot(_dvh_info_fake[0],_dvh_info_fake[1],linewidth=2,color=colors[idx],linestyle=':',label=organ_labels[organ_name]+" Fake")
                # label=_dvh_info_fake[2],
            ax.set_xlabel("Dose(Gy)")
            ax.set_ylabel("Ratio of total structure volume(%)")

            plt.legend(prop={'size': 6})
            plt.title("Dose-Volume Histogram")
            plt.savefig(f"./{self.pname}_DVH.png")
            plt.show()
            if self.redundancy:
                print("DVH info:")
                print(self.D_info)
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
    dvh.plot_dvh()
