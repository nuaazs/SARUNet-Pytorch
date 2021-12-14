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

class DVH(object):
    """Get DVH from the dose results of MC."""


    def __init__(self,MC_out_file_pcath,organs_list):
        """ Init fun. Get `organs_array_dict` and xxx_dose array.

        Args:
            MC_out_file_pcath ([str]): [The path to the output file]
            organs_list ([str list]): [organ's name list]
        """
        
        self.organs = organs_list
        self.organs_array_dict = {}
        self.organs_dvh_dict = {}
        self.organs_path = "./organs/"
        for organ in organs_list:
            self.organs_array_dict[organ] = self._get_organ_array(self.organ)
        self.total_dose, self.boron_dose, self.fast_n_dose, self.N_dose = self._get_dose_array(MC_out_file_pcath)
    
    
    def _array_from_MC(self,MC_OUT_FILE):
        """Get a two-dimensional numpy array from the output file 
           of Monte Carlo software.

        Args:
            MC_OUT_FILE ([str]): [The path to the output file]
        Returns:
            [array]: [The result of this output file in a two-dimensional array format]
        """
        # with open(file,encoding = 'utf-8') as f:
        #     data = np.loadtxt(f,delimiter = ",", skiprows = 8)
        # #return data
        # output = data[:,3].reshape(256,256,187).transpose(2,0,1)
        # filename = file.split("/")[-1].split(".")[-2]
        # if "fake" in file:
        #     np.save("./fake_out/"+filename+".npy",output)
        # elif "real" in file:
        #     np.save("./real_out/"+filename+".npy",output)
        # print(f"{filename} Done.")
        # return 0
        pass


    def _get_organ_array(self,organ):
        """Get a mask for the specified organ.

        Args:
            filepath ([str]): [organ arrays saved path]
            organ ([str]): [The name of the organ]
        Returns:
            [array]: [mask for the specified organ]
        """
        return self._get_array(self.organs_path,organ)



    def _get_dose_array(self,outpath):
        """Get total dose array, and Boron/fast n/gamma/nitrogen dose.

        Args:
            outpath ([str]): [ ([str]): [The path to the MC output file]]

        Returns:
            [array]: [ Total dose array for BNCT]
        """
        r=36
        n=6.5
        xx = n*r*3.1415926
        t=50 #tumour
        s=25 #skin
        nt=18#NT Boron concentration
        self.boron_array = self._array_from_MC(os.path.join(outpath,"boron.txt"))
        self.fast_array = self._array_from_MC(os.path.join(outpath,"fast.txt"))
        self.gamma_array = self._array_from_MC(os.path.join(outpath,"gamma.txt"))
        self.nitrogen_array = self._array_from_MC(os.path.join(outpath,"nitrogen.txt"))
        self.ct = self._get_organ_array("ct")
        self.gtv = self._get_organ_array("gtv")
        total_dose= np.zeros(self.ct.shape)
        total_dose[self.ct<-999] = self.boron_array[self.ct<-999]*xx* \
                        + self.fast_array[self.ct<-999]*xx \
                        + self.gamma_array[self.ct<-999]*xx \
                        + self.nitrogen_array[self.ct<-999]*xx
        total_dose = self.boron_array*xx*nt \
                    + self.fast_array*xx \
                    + self.gamma_array*xx \
                    + self.nitrogen_array*xx
        
        total_dose[self.gtv==1] = self.boron_array[self.gtv==1]*xx*t \
                    + self.fast_array[self.gtv==1]*xx \
                    + self.gamma_array[self.gtv==1]*xx \
                    + self.nitrogen_array[self.gtv==1]*xx
        return total_dose


    def _get_dvh_data(self):
        """Get dvh information for all organs and save it in a dictionary.
        """
        for organ_name in self.organs_array_dict.keys():
            xy_list = self._get_organ_dvh(self.organs_array_dict[organ_name])
            xy_list.append(organ_name) ## [xx,yy,label]
            self.organs_dvh_dict[organ_name] = xy_list
        return 0

    def _get_organ_dvh(self,organ_array):
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
        out_dose[organ_array>0] = self.total_dose[organ_array>0]
        max_dose = self.total_dose.max()
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
        for organ_name in self.organs_array_dict.keys():
            _dvh_info = self.organs_dvh_dict[organ_name]
            plt.plot(_dvh_info[0],_dvh_info[1],label=_dvh_info[2],linewidth=1)
        plt.legend()
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
    mcnp_output_path = "./output"
    dvh = DVH(mcnp_output_path,["eye_l","eye_r","gtv","brain","skull"])
    
    plt.figure()
    plt.plot(dvh.total_dose)
    plt.title("Total Dose")
    plt.axis('off')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(dvh.boron_dose)
    plt.title("Boron Dose")
    plt.axis('off')
    plt.colorbar()
    plt.show()

    dvh.plot(dvh)