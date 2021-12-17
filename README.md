# Attn-UNet: A deep learning method for generating virtual CT from MRI for dose calculation of BNCT

<div align=center>
<img src="https://img.shields.io/badge/Pytorch-1.10.1-green.svg"  />
<img src="https://img.shields.io/badge/Python-3.9-blue.svg"  />
<img src="https://img.shields.io/badge/CBAM-green.svg"  />
<img src="https://img.shields.io/badge/Unet-pink.svg"  />
<img src="https://img.shields.io/badge/MRI-CT-red.svg"  />
</div>

- [Attn-UNet: A deep learning method for generating virtual CT from MRI for dose calculation of BNCT](#attn-unet-a-deep-learning-method-for-generating-virtual-ct-from-mri-for-dose-calculation-of-bnct)
  - [Preparation](#preparation)
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

All the models were trained on 2\*NVIDIA 12GB TITAN V.



### Environment setup

We advise the creation of a new conda environment including all necessary packages. The repository includes a requirements file. Please create and activate the new environment with

```
conda env create -f requirements.yml
conda activate attngan
```



### Dataset preparation

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



## Pretrained weights

We release a pretrained set of weights to allow reproducibility of our results. The weights are downloadable from [here](). Once downloaded, unpack the file in the root of the project and test them with the inference notebook.



## Training

The training routine of Attn-GAN is mainly based on the pix2pix codebase, available with details in the official repository.

To launch a default training, run

```shell
python train.py --data_root path/to/data --gpu_ids 0,1,2 --netG attnunet --netD basic --model pix2pix --name attnunet-gan
```



## MAE Result

|             | SIZE    | MEAN MAE | STD   | P1     | P2     | P3     | P4     | P5    | P6     | P7     | P8     | P9     | P10    | P11    |
| ----------- | ------- | -------- | ----- | ------ | ------ | ------ | ------ | ----- | ------ | ------ | ------ | ------ | ------ | ------ |
| Unet Basic  |         | 122.98   | 37.35 | 128.34 | 67.84  | 114.92 | 112.87 | 72.68 | 129.47 | 117.37 | 156.94 | 114.34 | 124.1  | 213.96 |
| Unet 128    | 41.823M | 72.88    | 16.99 | 78.21  | 33.69  | 82.38  | 75.91  | 65.57 | 55.79  | 65.69  | 86.99  | 83.6   | 101.12 | 72.67  |
| Vnet        |         |          |       |        |        |        |        |       |        |        |        |        |        |        |
| Pix2Pix     |         | 89.95    | 21.05 | 99.25  | 58.45  | 85.96  | 84.11  | 69.6  | 68.81  | 81.97  | 105.48 | 92.78  | 104.47 | 138.52 |
| DenseUnet   | 49.518  | 130.76   | 33.68 | 141.6  | 135.38 | 90.63  | 120.42 | 68.05 | 118.58 | 123.39 | 170.87 | 132.6  | 136.5  | 200.34 |
| ResUNet     |         |          |       |        |        |        |        |       |        |        |        |        |        |        |
| VggUNet     |         |          |       |        |        |        |        |       |        |        |        |        |        |        |
| Vgg19UNet   |         |          |       |        |        |        |        |       |        |        |        |        |        |        |
| SE_ResUNet  |         |          |       |        |        |        |        |       |        |        |        |        |        |        |
| AttnResUNet |         |          |       |        |        |        |        |       |        |        |        |        |        |        |
| AttnUNet    | 41.848M |          |       |        |        |        |        |       |        |        |        |        |        |        |
| AttnUNet DS | 37.113M | 68.48    | 16.13 | 71.96  | 35.47  | 62.65  | 70.16  | 56.48 | 50.11  | 67.79  | 89.82  | 85.38  | 90.45  | 73.07  |

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/L2bHi.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BLtPk.jpg)


## CBAM MODELS

![pic1](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/图片8.png)

### Spatial Attention

![spatial-attn](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/空间注意力机制.png)

### Channel Attention

![channel-attention](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/通道注意力机制.png)