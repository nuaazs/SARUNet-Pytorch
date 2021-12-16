# Attn-GAN: A deep learning method for generating virtual CT from MRI for dose calculation of BNCT



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
|              | P1     | P2    | P3     | P4     | P5     | P6    | P7     | P8     | P9     | P10    | P11    | P12    | P13    | MEAN   | STD  |
| ------------ | ------ | ----- | ------ | ------ | ------ | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---- |
| UNet         | 128.34 | 67.84 | 114.93 | 112.87 | 161.81 | 99.94 | 72.628 | 129.48 | 17.371 | 156.94 | 114.35 | 124.10 | 213.69 | 124.19 |      |
| ResNet       | 73.06  | 45.29 | 63.64  | 72.32  | 83.07  | 67.05 | 57.93  | 52.32  | 70.13  | 90.92  | 82.57  | 88.27  | 80.06  | 71.27  |      |
| AttnUNet     |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| Pix2Pix      | 99.25  | 58.45 | 85.97  | 84.12  | 92.28  | 84.71 | 69.60  | 68.81  | 81.97  | 105.48 | 92.78  | 104.47 | 138.52 | 89.72  |      |
| AttnUNet DS  | 71.96  | 35.47 | 62.65  | 70.16  | 87.87  | 78.72 | 56.48  | 50.112 | 67.79  | 89.82  | 85.38  | 90.45  | 73.07  | 70.76  |      |
| DenseUnet    |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| ResUnet      |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| CBAM_Resunet |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| VggUNet      |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| Vgg19UNet    |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| SE_ResUnet   |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |
| VNet         |        |       |        |        |        |       |        |        |        |        |        |        |        |        |      |

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/L2bHi.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BLtPk.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BmDPs.jpg)





## CBAM MODELS

![pic1](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/图片8.png)

### Spatial Attention

![spatial-attn](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/空间注意力机制.png)

### Channel Attention

![channel-attention](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/通道注意力机制.png)