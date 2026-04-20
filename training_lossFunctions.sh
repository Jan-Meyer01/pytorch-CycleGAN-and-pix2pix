# Trainings script for the different loss functions.

# default MSE loss for GAN loss in pix2pix
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lsgan
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lsgan

# LPIPS as a loss
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpips_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lpips
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpips_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lpips

# sharpness (gradient) loss
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_sharp_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode sharp
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_sharp_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode sharp

# MSE + sharpness loss
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_mseSharp_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode mseSharp
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_mseSharp_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode mseSharp

# LPIPS + sharpness loss
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpipsSharp_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lpipsSharp
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpipsSharp_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lpipsSharp