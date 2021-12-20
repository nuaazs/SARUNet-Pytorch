# SARU-Net: A Self Attention ResUnet to generate synthetic CT images for MRI-only BNCT treatment planning

<div align=center>
<img src="https://img.shields.io/badge/Pytorch-1.10.1-green.svg"  />
<img src="https://img.shields.io/badge/Python-3.9-blue.svg"  />
<img src="https://img.shields.io/badge/CBAM-green.svg"  />
<img src="https://img.shields.io/badge/ResUNet-pink.svg"  />
<img src="https://img.shields.io/badge/MRI-CT-red.svg"  />
<img src="https://img.shields.io/badge/MCNP-6.0-blue.svg"  />
<img src="https://img.shields.io/badge/BNCT-Dose Calc-green.svg"  />
<img src="https://img.shields.io/badge/IINT-red.svg"  />
</div>
<div>
<br>
<br>
</div>
<div align=center>
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/saru_mm_压缩后.png" width="800px" />
</div>



> 该项目还未完成。
>
> Note: The project is not yet complete.



## Todo List:

- SARU++
- SARU
- VNet/Unet/Resnet/pix2pix



## Table of Content


- [SARU-Net: A Self Attention ResUnet to generate synthetic CT images for MRI-only BNCT treatment planning](#saru-net-a-self-attention-resunet-to-generate-synthetic-ct-images-for-mri-only-bnct-treatment-planning)
  - [Todo List:](#todo-list)
  - [Table of Content](#table-of-content)
  - [Preparation](#preparation)
    - [Environment setup](#environment-setup)
    - [Dataset preparation](#dataset-preparation)
  - [Pretrained weights](#pretrained-weights)
  - [Training](#training)
  - [MAE Result of 13 patients](#mae-result-of-13-patients)
  - [CBAM MODELS](#cbam-models)
    - [Spatial Attention](#spatial-attention)
    - [Channel Attention](#channel-attention)
    - [Attention ResBlock](#attention-resblock)
  - [Code structure](#code-structure)
  - [Pull Request](#pull-request)
  - [Citation](#citation)
  - [Other Languages](#other-languages)
  - [Related Projects](#related-projects)
  - [Acknowledgments](#acknowledgments)
    
    - [Environment setup](#environment-setup)
    - [Dataset preparation](#dataset-preparation)
  - [Pretrained weights](#pretrained-weights)
  - [Training](#training)
  - [MAE Result](#mae-result)
  - [CBAM MODELS](#cbam-models)
    
    - [Spatial Attention](#spatial-attention)
    
    - [Channel Attention](#channel-attention)
    
      


## Preparation

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN



### Environment setup

We advise the creation of a new conda environment including all necessary packages. The repository includes a requirements file. Please create and activate the new environment with

```
conda env create -f requirements.yml
conda activate attngan
```



### Dataset preparation

<div align=center>
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/preprocess_压缩后.png" width="600px" />
</div>

Running those commands should result in a similar directory structure:

```
root
  datasets
    MRICT
      train
          patient_001_001.png
          ...
          patient_002_001.png
          ...
		  patient_100_025.png
      test
      	  patient_101_001.png
      	  ...
          patient_102_002.png
          ...
		  patient_110_025.png
      val
          patient_111_001.png
          ...
          patient_112_002.png
          ...
          ...
```

Our pre-trained model used **130 + patient** cases, for a total of about 4500 image pairs, while performing data enhancement methods such as random flipping, random scaling, and random cropping.



## Pretrained weights

We release a pretrained set of weights to allow reproducibility of our results. The weights are downloadable from [Google Drive]()(or [百度云]()). Once downloaded, unpack the file in the root of the project and test them with the inference notebook.

All the models were trained on 2\*NVIDIA 12GB TITAN V.



## Training

The training routine of Attn-GAN is mainly based on the pix2pix codebase, available with details in the official repository.

To launch a default training, run

```shell
python train.py --data_root path/to/data --gpu_ids 0,1,2 --netG attnunet --netD basic --model pix2pix --name attnunet-gan
```



## MAE Result of 13 patients

| BackBone            | Params      | MEAN MAE  | STD       | MEAN ME  | STD      | MEAN RMSE  | STD       |
| ------------------- | ----------- | --------- | --------- | -------- | -------- | ---------- | --------- |
| UNet                | 17.266 M    | 124.2     | 36.56     | 64.07    | 30.77    | 338.66     | 68.39     |
| ResNet              | 11.371M     | 71.28     | 13.34     | -3.27    | 8.48     | 202.82     | 37.78     |
| DeepUNet            | 41.823M     | 73.8      | 15.77     | -2.2     | 13.35    | 212.8      | 40.86     |
| Pix2Pix             | 44.588M     | 89.72     | 19.42     | 5.07     | 16.68    | 238.08     | 42.47     |
| DenseUNet           | 49.518M     | 130.32    | 31.42     | 58.57    | 33.8     | 348.42     | 73.07     |
| **SARU-Net (ours)** | **16.212M** | **62.61** | **11.26** | **9.77** | **7.84** | **183.04** | **30.26** |

<div align=center>
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/MAE_压缩后.png" width="600px" />
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BLtPk.jpg" width="600px" />
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/L2bHi.jpg" width="600px" />
</div>


## CBAM MODELS
<div align=center>
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/conv_pool_attn_压缩后.png" width="600px" />
</div>


### Spatial Attention
<div align=center>
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/spatial_压缩后.png" width="600px" />
</div>


### Channel Attention
<div align=center>
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/channel_压缩后.png" width="600px" />
</div>
### Attention ResBlock






## Code structure

To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module [here](https://iint.icu/).



## Pull Request

You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/). Please run `flake8 --ignore E501 .` and `python ./scripts/test_before_push.py` before you commit the code. Please also update the code structure [overview](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/overview.md) accordingly if you add or remove files.



## Citation

If you use this code for your research, please cite our papers.

```
@inproceedings{}
}
```



## Other Languages

[简体中文]()



## Related Projects

**[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)**
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) | [pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)| [BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**



## Acknowledgments

Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pytorch-CBAM](https://github.com/luuuyi/CBAM.PyTorch)
