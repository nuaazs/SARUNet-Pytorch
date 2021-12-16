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
|              | P1          | P2          | P3          | P4   | P5   | P6   | P7   | P8   | P9   | P10  |
| ------------ | ----------- | ----------- | ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| UNet         | 66.91847405 | 75.7184181  | 51.22233685 |      |      |      |      |      |      |      |
| ResNet       | 62.97917012 | 74.29159858 | 48.00514584 |      |      |      |      |      |      |      |
| AttnUNet     | 59.32602055 | 68.93765585 | 39.55022302 |      |      |      |      |      |      |      |
| Pix2Pix      |             |             |             |      |      |      |      |      |      |      |
| VNet         |             |             |             |      |      |      |      |      |      |      |
| DenseUnet    |             |             |             |      |      |      |      |      |      |      |
| ResUnet      |             |             |             |      |      |      |      |      |      |      |
| CBAM_Resunet |             |             |             |      |      |      |      |      |      |      |
| VggUNet      |             |             |             |      |      |      |      |      |      |      |
| Vgg19UNet    |             |             |             |      |      |      |      |      |      |      |
| SE_ResUnet   |             |             |             |      |      |      |      |      |      |      |

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/L2bHi.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BLtPk.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BmDPs.jpg)





## CBAM MODELS

![pic1](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/图片8.png)

### Spatial Attention

![spatial-attn](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/空间注意力机制.png)

### Channel Attention

![channel-attention](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/通道注意力机制.png)