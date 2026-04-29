# Trainings and test scripts for different generator architectures and loss functions.

# gradient loss (sharpness) for GAN loss in pix2pix with 
#resnet_9blocks as generator
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet9_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode sharp --netG resnet_9blocks
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet9_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode sharp --netG resnet_9blocks

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet9_AtoB --model pix2pix --direction AtoB --num_test 120 --netG resnet_9blocks
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet9_BtoA --model pix2pix --direction BtoA --num_test 120 --netG resnet_9blocks

# resnet_6blocks as generator
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet6_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode sharp --netG resnet_6blocks
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet6_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode sharp --netG resnet_6blocks

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet6_AtoB --model pix2pix --direction AtoB --num_test 120 --netG resnet_6blocks
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_resnet6_BtoA --model pix2pix --direction BtoA --num_test 120 --netG resnet_6blocks

# unet_128 as generator 
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet128_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode sharp --netG unet_128
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet128_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode sharp --netG unet_128

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet128_AtoB --model pix2pix --direction AtoB --num_test 120 --netG unet_128
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet128_BtoA --model pix2pix --direction BtoA --num_test 120 --netG unet_128

# unet_256 as generator 
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet256_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode sharp --netG unet_256
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet256_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode sharp --netG unet_256

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet256_AtoB --model pix2pix --direction AtoB --num_test 120 --netG unet_256
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_sharp_unet256_BtoA --model pix2pix --direction BtoA --num_test 120 --netG unet_256


# LPIPS loss for GAN loss in pix2pix with 
#resnet_9blocks as generator
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet9_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lpips --netG resnet_9blocks
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet9_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lpips --netG resnet_9blocks

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet9_AtoB --model pix2pix --direction AtoB --num_test 120 --netG resnet_9blocks
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet9_BtoA --model pix2pix --direction BtoA --num_test 120 --netG resnet_9blocks

# resnet_6blocks as generator
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet6_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lpips --netG resnet_6blocks
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet6_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lpips --netG resnet_6blocks

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet6_AtoB --model pix2pix --direction AtoB --num_test 120 --netG resnet_6blocks
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_resnet6_BtoA --model pix2pix --direction BtoA --num_test 120 --netG resnet_6blocks

# unet_128 as generator
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet128_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lpips --netG unet_128
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet128_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lpips --netG unet_128

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet128_AtoB --model pix2pix --direction AtoB --num_test 120 --netG unet_128
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet128_BtoA --model pix2pix --direction BtoA --num_test 120 --netG unet_128

# unet_256 as generator
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet256_AtoB --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lpips --netG unet_256
python train.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet256_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lpips --netG unet_256

python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet256_AtoB --model pix2pix --direction AtoB --num_test 120 --netG unet_256
python test.py --dataroot ./datasets/EPI_modelling_SE --name EPI_modelling_SE_pix2pix_lpips_unet256_BtoA --model pix2pix --direction BtoA --num_test 120 --netG unet_256
