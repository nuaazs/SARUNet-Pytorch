# Attn-GAN: A deep learning method for generating virtual CT from MRI for dose calculation of BNCT

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/L2bHi.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BLtPk.jpg)
![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/BmDPs.jpg)

## Preparation

All the models have been trained on 2\*12GB GTX1080Ti GPU.

### Environment setup

We advise the creation of a new conda environment including all necessary packages. The repository includes a requirements file. Please create and activate the new environment with

```
conda env create -f requirements.yml
conda activate attngan
```

### Dataset preparation



## Pretrained weights

We release a pretrained set of weights to allow reproducibility of our results. The weights are downloadable from [here](). Once downloaded, unpack the file in the root of the project and test them with the inference notebook.

## Training

The training routine of Attn-GAN is mainly based on the pix2pix codebase, available with details in the official repository.

To launch a default training, run

```
python train.py --path_data path/to/data --gpu_ids 0
```



## MAE Result
|                  | Patient1    | Patient2    | Patient3    |
| ---------------- | ----------- | ----------- | ----------- |
| Unet256-GAN      | 66.91847405 | 75.7184181  | 51.22233685 |
| Resnet9block-GAN | 62.97917012 | 74.29159858 | 48.00514584 |
| Attn-GAN         | 59.32602055 | 68.93765585 | 39.55022302 |
