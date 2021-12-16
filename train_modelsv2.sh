python train.py --netG denseunet --mode generate --name bs10_denseunet --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG unetv2 --mode generate --name bs10_unetv2 --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resunetv2 --mode generate --name bs10_resunetv2 --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG se_resunet --mode generate --name bs10_se_resunet  --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG cbamresunet --mode generate --name bs10_cbamresunet  --display_port 8097 --gpu_ids 1 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \




python train.py --netG dilatedresunet --mode generate --name bs10_dilatedresunet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG vggunet --mode generate --name bs10_vggunet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG vgg19unet --mode generate --name bs10_vgg19unet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG vnet --mode generate --name bs10_vnet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG VFN --mode generate --name bs10_VFN  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10
