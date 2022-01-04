# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/")
from util.DVH import DVH

for pname in ["001","geng03_007","geng03_010","038","geng03_051","052","geng03_050","geng03_009"]:
    mcnp_output_path = f"/home/zhaosheng/paper2/online_code/cbamunet-pix2pix/results/SARUp_dream_3/dose/{pname}/"
    dvh = DVH(mcnp_output_path,pname=pname,["gtv"],redundancy=False) #,"lens_r","gtv","brain","skull"
    print(f"Total Dose Shape:{dvh.total_dose_real.shape}")
    dvh.plot_dvh(reload=True)
