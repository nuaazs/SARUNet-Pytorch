
# 37.130M
python test.py --netG attnunet_128_ds --epoch best --mode pix2pix --name bs10_attnunet_128_ds_basic  --gpu_ids 1 --direction AtoB \
&& \

# 
python test.py --netG unet_128 --epoch best --mode pix2pix --name bs10_unet_128_basic  --gpu_ids 1 --direction AtoB \
&& \

# 
python test.py --netG attnunet_128 --epoch best --mode pix2pix --name bs10_attnunet_128_basic  --gpu_ids 1 --direction AtoB \
&& \

# 47.654M
python test.py --netG attnunet_256_ds --epoch best --mode pix2pix --name bs10_attnunet_256_ds_basic  --gpu_ids 1 --direction AtoB \
&& \

# 54.408M
python test.py --netG unet_256 --epoch best --mode pix2pix --name bs10_unet_256_basic  --gpu_ids 1 --direction AtoB \
&& \

# 54.442M
python test.py --netG attnunet_256 --epoch best --mode pix2pix --name bs10_attnunet_256_basic  --gpu_ids 1 --direction AtoB \
&& \

# 11.371M
python test.py --netG resnet --epoch best --mode pix2pix --name bs10_resnet_basic  --gpu_ids 1 --direction AtoB \
&& \

# 11.371M
python test.py --netG attnresnet --epoch best --mode pix2pix --name bs10_attnresnet_basic  --gpu_ids 1 --direction AtoB \
&& \


# python test.py --netG resunet --epoch best --mode pix2pix --name bs10_resunet_basic  --gpu_ids 1 --direction AtoB \
# && \


# python test.py --netG resunetpp --epoch best --mode pix2pix --name bs10_resunetpp_basic  --gpu_ids 1 --direction AtoB \
# && \




################################################################################################



# 37.113M
python test.py --mode pix2pix --netG attnunet_128_ds --epoch best --name bs10_attnunet_128_ds --gpu_ids 0 --direction AtoB \
&& \

# 41.823M
python test.py --mode pix2pix --netG unet_128 --epoch best --name bs10_unet_128 --gpu_ids 0 --direction AtoB \
&& \

# 41.848M
python test.py --mode pix2pix --netG attnunet_128 --epoch best --name bs10_attnunet_128 --gpu_ids 0 --direction AtoB \
&& \

# 47.654M
python test.py --mode pix2pix --netG attnunet_256_ds --epoch best --name bs10_attnunet_256_ds --gpu_ids 0 --direction AtoB \
&& \

# 54.408M
python test.py --mode pix2pix --netG unet_256 --epoch best --name bs10_unet_256 --gpu_ids 0 --direction AtoB \
&& \

# 54.442M
python test.py --mode pix2pix --netG attnunet_256 --epoch best --name bs10_attnunet_256 --gpu_ids 1 --direction AtoB \
&& \

# 11.371M
python test.py --mode pix2pix --netG resnet --epoch best --name bs10_resnet  --gpu_ids 1 --direction AtoB \
&& \

# 11.371M
python test.py --mode pix2pix --netG attnresnet --epoch best --name bs10_attnresnet  --gpu_ids 1 --direction AtoB \
&& \

# 13.041
python test.py --mode pix2pix --netG resunet --epoch best --name bs10_resunet  --gpu_ids 1 --direction AtoB
# && \

# python test.py --netG resunetpp --epoch best --name bs10_resunetpp  --gpu_ids 0 --direction AtoB
