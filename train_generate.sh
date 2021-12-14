python train.py --netG attnunet_128_ds --mode generate --name bs10_attnunet_128_ds --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG unet_128 --mode generate --name bs10_unet_128 --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet_128 --mode generate --name bs10_attnunet_128 --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet_256_ds --mode generate --name bs10_attnunet_256_ds --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG unet_256 --mode generate --name bs10_unet_256 --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnunet_256 --mode generate --name bs10_attnunet_256 --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resnet --mode generate --name bs10_resnet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG attnresnet --mode generate --name bs10_attnresnet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resunet --mode generate --name bs10_resunet  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10\
&& \
python train.py --netG resunetpp --mode generate --name bs10_resunetpp  --display_port 8097 --gpu_ids 0 --display_id 0  --save_epoch_freq 50 --print_freq 1000 --batch_size 10
