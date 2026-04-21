# Train and test a network after one epoch as a baseline.

# default MSE loss for GAN loss in pix2pix
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_AtoB_baseline --model pix2pix --direction AtoB --load_size 256 --print_freq 1000 --gan_mode lsgan --n_epochs 1 --save_epoch_freq 1 --n_epochs_decay 0
python train.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_BtoA_baseline --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --gan_mode lsgan --n_epochs 1 --save_epoch_freq 1 --n_epochs_decay 0

python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_AtoB_baseline --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_BtoA_baseline --model pix2pix --direction BtoA --num_test 120