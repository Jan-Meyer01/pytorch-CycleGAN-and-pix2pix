# Trainings and test scripts for DWI experiments

# gradient loss for GAN loss in pix2pix with resnet_9blocks as generator
python train.py --dataroot ./datasets/DWI --name DWI_pix2pix_grad_resnet9_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 5000 --n_epochs 50 --n_epochs_decay 50 --netG resnet_9blocks --batch_size 30

# test performance on synthetic test set
python test.py --dataroot ./datasets/DWI --name DWI_pix2pix_grad_resnet9_BtoA --model pix2pix --direction BtoA --num_test 4440 --netG resnet_9blocks

# TODO: load trained network, fine-tune for x epochs using synthetic test set and test on real data!
python train.py --dataroot ./datasets/DWI --name DWI_pix2pix_grad_resnet9_BtoA --model pix2pix --direction BtoA --load_size 256 --print_freq 1000 --n_epochs 0 --n_epochs_decay 15 --netG resnet_9blocks --suffix  --epoch 100 --continue_train