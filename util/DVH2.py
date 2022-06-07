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
# @File    : util/DVH.py
# @Describe: Get DVH from the dose results of MC(mcnp/topas/G4).

import matplotlib.pyplot as plt
import scipy.io as scio
import nibabel as nb
import numpy as np
import os
import ants

class DVH(object):
    """Get DVH from the dose results of MC."""
    def __init__(self,
                 
                 # 剂量计算结果路径，即csv所在的文件夹
                 MC_out_file_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/dose_result/",
                 
                 # 考虑的器官
                 organs_list=["len_l","len_r","gtv","brain","brainstem","skull","on_l","on_r","eye_l","eye_r","skin"],
                 
                 # 所有病人的所有器官mask npy文件所在的地方
                 organs_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/seg/",
                 
                 # 病人编号
                 pname="003",
                 
                 # 文件类型，默认csv
                 filetype="csv",
                 #shape=(256,256,180),
                 
                 # 是否打印中间过程，默认否
                 redundancy=False,
                 
                 # 真实ct的nii路径
                 real_ct_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/real_ct_niis_0318",
                 
                 cache_path="/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/dose_result/cache/",
                 
                 reload=False,
                 plot_boron_concentration = True
                 ):
        """ Init fun. Get `organs_array_dict` and xxx_dose array.

        Args:
            MC_out_file_path ([str]): [The path to the output file]
            organs_list ([str list]): [organ's name list]
        """
        self.organs = organs_list
        self.pname = pname
        self.organs_array_dict = {}
        self.organs_dvh_dict_real = {}
        self.organs_dvh_dict_fake = {}
        self.redundancy = redundancy
        self.organs_path = organs_path
        self.real_ct_path = real_ct_path
        self.D_info = {}
        self.reload = reload
        self.plot_boron_concentration = plot_boron_concentration
        
        
        self.fake_cache = os.path.join(cache_path,f"fake/{self.pname}")#os.path.join(cache_path,f"fake/fake_dvh.npy")
        self.real_cache = os.path.join(cache_path,f"real/{self.pname}")#os.path.join(cache_path,f"real/{self.pname}/real_dvh.npy")
        self.dose_cache = os.path.join(cache_path,f"dose/{self.pname}")#os.path.join(cache_path,f"real/{self.pname}/dvh.npy")
        os.makedirs(self.fake_cache,exist_ok=True)
        os.makedirs(self.real_cache,exist_ok=True)
        os.makedirs(self.dose_cache,exist_ok=True)
        
        self.raw_img = ants.image_read(os.path.join(self.real_ct_path,f"{pname}.nii"))
        self.ct = self.raw_img.numpy().swapaxes(0,1)

        for organ in organs_list:
            if self.redundancy:
                print(f"Now {organ}.")
            self.organs_array_dict[organ] = self._get_organ_array(organ,self.pname)
        if filetype=="csv":
            if self.reload or (not (os.path.exists(os.path.join(self.dose_cache,f"{pname}_total_dose_fake.nii")))):
                self.total_dose_real,self.total_dose_fake = self._get_dose_array_from_csv(MC_out_file_path,self.pname)

                self.total_dose_real_img = ants.from_numpy(self.total_dose_real)
                self.total_dose_real_img.set_spacing(self.raw_img.spacing)
                ants.image_write(self.total_dose_real_img,os.path.join(self.dose_cache,f"{pname}_total_dose_real.nii"))

                self.total_dose_fake_img = ants.from_numpy(self.total_dose_fake)
                self.total_dose_fake_img.set_spacing(self.raw_img.spacing)
                ants.image_write(self.total_dose_fake_img,os.path.join(self.dose_cache,f"{pname}_total_dose_fake.nii"))
            else:
                self.total_dose_fake_img = ants.image_read(os.path.join(self.dose_cache,f"{pname}_total_dose_fake.nii"))
                self.total_dose_fake = self.total_dose_fake_img.numpy()
                self.total_dose_real_img = ants.image_read(os.path.join(self.dose_cache,f"{pname}_total_dose_real.nii"))
                self.total_dose_real = self.total_dose_real_img.numpy()
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
            if (not self.reload) and os.path.exists(MC_OUT_FILE[:-4]+".npy"):
                filename_ = MC_OUT_FILE[:-4]+".npy"
                # if self.redundancy:
                print(f"\t->Loading npy file : {filename_}")
                array_ = np.load(filename_)
                # if self.redundancy:
                print(f"\t->Shape:{array_.shape}")
                return array_
            # if self.redundancy:
            print(f"\t->Reading MC output file : {MC_OUT_FILE}")
            with open(MC_OUT_FILE,encoding = 'utf-8') as f:
                data = np.loadtxt(f,delimiter = ",", skiprows = 8)
            output = data[:,3].reshape(256,256,-1)
            np.save(MC_OUT_FILE[:-4],output)
            if self.redundancy:
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
        return self._get_array(self.organs_path,pname+"_"+organ).swapaxes(0,1)

    def _get_dose_array_from_csv(self,outpath,pname):
        """Get total dose array, and Boron/fast n/gamma/nitrogen dose.

        Args:
            outpath ([str]): [ ([str]): [The path to the MC output file]]

        Returns:
            [array]: [ Total dose array for BNCT]
        """
        r=36
        n=6.5
        xx = n*r*3.1415926*30*60 # 30 min
        t=60 #tumour
        s=25 #skin
        nt=18#NT Boron concentration
        self.boron_array = self._array_from_MC(os.path.join(outpath,"boron10.csv"))
        self.fast_array = self._array_from_MC(os.path.join(outpath,"fast10.csv"))
        self.gamma_array = self._array_from_MC(os.path.join(outpath,"gamma10.csv"))
        self.nitrogen_array = self._array_from_MC(os.path.join(outpath,"nitrogen10.csv"))


        self.gtv = self._get_organ_array("gtv",self.pname)
        self.skin = self._get_organ_array("skin",self.pname)
        self.len_l = self._get_organ_array("len_l",self.pname)
        self.len_r = self._get_organ_array("len_r",self.pname)


        boron_concentration = np.zeros(self.ct.shape)+18*1.4
        if self.redundancy:
            print(f"Boron concentration shape: {boron_concentration.shape}")
        
        boron_concentration[self.ct<-999]=0 # 空气
        boron_concentration[self.len_r==1]=10*1.4
        boron_concentration[self.len_l==1]=10*1.4
        boron_concentration[self.skin==1]=25*2.5
        boron_concentration[self.gtv==1]=60*3.8#60*3.8
    

        if self.plot_boron_concentration:
            plt.figure()
            plt.imshow(self.boron_array_real[:,:,10])
            plt.title("Boron Dose real")
            plt.show()

            plt.figure()
            plt.imshow(boron_concentration[:,:,10])
            plt.title("Boron concentration")
            plt.show()
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

    def plot_dvh(self,save_fig_path="none",show_pic=False):
        
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
          "peru",
          "gold",
          "olive"]

        organ_labels  = {"len_l":"Lens(L)",
                 "len_r":"Lens(L)",
                 "on_l":"ON(L)",
                 "on_r":"ON(R)",
                 "brain":"Brain",
                 "brainstem":"BrainStem",
                 "gtv":"GTV",
                 "skull":"Skull",
                 "soft_tissue":"Soft tissue",
                 "bone":"Bone",
                 "air":"Air",
                 "skin":"Skin",
                 "eye_l":"Eye(L)",
                 "eye_r":"Eye(R)"}

        plt.style.use('ggplot')
        fg=plt.figure(dpi=400)
        ax=fg.add_subplot(1,1,1)
        real_dvh_cache_path = os.path.join(self.real_cache,"real_dvh.npy")
        fake_dvh_cache_path = os.path.join(self.fake_cache,"fake_dvh.npy")
        dvh_cache_path = os.path.join(self.dose_cache,"dvh.npy")
        if self.reload or (not (os.path.exists(real_dvh_cache_path))):
            self._get_dvh_data(self.total_dose_real,mode="real")
            np.save(real_dvh_cache_path, self.organs_dvh_dict_real)
        
        
        if self.reload or (not (os.path.exists(fake_dvh_cache_path))):
            self._get_dvh_data(self.total_dose_fake,mode="fake")
            np.save(fake_dvh_cache_path, self.organs_dvh_dict_fake)
            
            
        if self.reload or (not (os.path.exists(dvh_cache_path))):
            self._get_dvh_data(self.total_dose_real,mode="real")
            self._get_dvh_data(self.total_dose_fake,mode="fake")
            np.save(real_dvh_cache_path, self.organs_dvh_dict_real)
            np.save(fake_dvh_cache_path, self.organs_dvh_dict_fake)
            np.save(dvh_cache_path, self.D_info)

        self.organs_dvh_dict_real = np.load(os.path.join(real_dvh_cache_path),allow_pickle=True).item()
        self.organs_dvh_dict_fake = np.load(os.path.join(fake_dvh_cache_path),allow_pickle=True).item()
        self.D_info = np.load(dvh_cache_path,allow_pickle=True).item()
        
        
        if show_pic:
            for idx,organ_name in enumerate(self.organs_array_dict.keys()):
                _dvh_info_real = self.organs_dvh_dict_real[organ_name]
                if max(_dvh_info_real[0])>200:
                    _dvh_info_real[0] = np.array(_dvh_info_real[0])/100.
                ax.plot(_dvh_info_real[0],_dvh_info_real[1],linewidth=1,color=colors[idx],linestyle='-',label=organ_labels[organ_name]+" Real")
                _dvh_info_fake = self.organs_dvh_dict_fake[organ_name]
                if max(_dvh_info_fake[0])>200:
                    _dvh_info_fake[0] = np.array(_dvh_info_fake[0])/100.
                ax.plot(_dvh_info_fake[0],
                        _dvh_info_fake[1],linewidth=1,
                        color=colors[idx],linestyle=':',
                        label=organ_labels[organ_name]+" Fake")
                # label=_dvh_info_fake[2],
            ax.set_xlabel("Dose(Gy)")
            ax.set_ylabel("Ratio of total structure volume(%)")

            plt.legend(prop={'size': 6})
            plt.title("Dose-Volume Histogram")
            if save_fig_path!="none":
                plt.savefig(os.path.join(save_fig_path,f"{self.pname}_DVH.png"))
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
    dvh = DVH(mcnp_output_path,["len_l","len_r","gtv","brain","skull"]) #
    if self.redundancy:
        print(f"Total Dose Shape:{dvh.total_dose_real.shape}")
    dvh.plot_dvh()
