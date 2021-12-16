python train.py --netG attnunet --netD basic --mode pix2pix --name bs10_attnunet_basic --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG unet --netD basic --mode pix2pix --name bs10_unet_basic --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnresnet --netD basic --mode pix2pix --name bs10_attnresnet_basic --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG smatunet --netD basic --mode pix2pix --name bs10_smatunet_basic  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10


python train.py --netG unet --mode generate --name bs10_unet --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet --mode generate --name bs10_attnunet --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnresnet --mode generate --name bs10_attnresnet --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG smatunet --mode generate --name bs10_smatunet  --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10
