python train.py --netG attnunet_128_ds --netD basic --mode pix2pix --name bs10_attnunet_128_ds_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG unet_128 --netD basic --mode pix2pix --name bs10_unet_128_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet_128 --netD basic --mode pix2pix --name bs10_attnunet_128_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet_256_ds --netD basic --mode pix2pix --name bs10_attnunet_256_ds_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG unet_256 --netD basic --mode pix2pix --name bs10_unet_256_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet_256 --netD basic --mode pix2pix --name bs10_attnunet_256_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resnet --netD basic --mode pix2pix --name bs10_resnet_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnresnet --netD basic --mode pix2pix --name bs10_attnresnet_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resunet --netD basic --mode pix2pix --name bs10_resunet_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resunetpp --netD basic --mode pix2pix --name bs10_resunetpp_basic --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
