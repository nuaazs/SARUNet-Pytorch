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
<img src="https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/stage_压缩后.png" width="800px" />
</div>


## Todo List:

- SARU++
- 3D SARU++


Backend:[SARU-flask](https://github.com/nuaazs/SARU-Flask)
Frontend:[SARU-VUE](https://github.com/nuaazs/mri2ct-vue)
## Table of Content

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


## Code structure

To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module [here](https://iint.icu/).



## Pull Request

You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/). Please run `flake8 --ignore E501 .` and `python ./scripts/test_before_push.py` before you commit the code. Please also update the code structure [overview](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/overview.md) accordingly if you add or remove files.



## Citation

If you use this code for your research, please cite our papers.

```
@article{zhao2022saru,
  title={SARU: A self attention ResUnet to generate synthetic CT images for MR-only BNCT treatment planning},
  author={Zhao, Sheng and Geng, Changran and Guo, Chang and Tian, Feng and Tang, Xiaobin},
  journal={Medical Physics},
  year={2022},
  publisher={Wiley Online Library}
}
```



## Related Projects

**[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)**
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) | [pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)| [BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**



## Acknowledgments

Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [pytorch-CBAM](https://github.com/luuuyi/CBAM.PyTorch)
