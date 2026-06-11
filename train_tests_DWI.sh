# Trainings and test scripts for DWI experiments

# gradient loss for GAN loss in pix2pix with resnet_9blocks as generator
python train.py --dataroot ./datasets/DWI_new --name DWI_new_pix2pix_grad_resnet9 --model pix2pix --direction BtoA --load_size 256 --print_freq 10000 --n_epochs 25 --n_epochs_decay 25 --netG resnet_9blocks --batch_size 30

# test performance on synthetic test set
python test.py --dataroot ./datasets/DWI_new --name DWI_new_pix2pix_grad_resnet9 --model pix2pix --direction BtoA --num_test 6660 --netG resnet_9blocks --batch_size 30 --epoch 100
mv ./results/DWI_pix2pix_grad_resnet9_BtoA/test_100 ./results/DWI_new_pix2pix_grad_resnet9/sub-tle005_Epoch_100

# load trained network and fine-tune for 15 epochs using synthetic test set
python train.py --dataroot ./datasets/DWI_new_sub-tle005 --name DWI_new_pix2pix_grad_resnet9 --model pix2pix --direction BtoA --load_size 256 --print_freq 5000 --n_epochs 5 --n_epochs_decay 10 --netG resnet_9blocks  --batch_size 30 --epoch 100 --continue_train

# re-test performance on synthetic test set for sanity check (it should have gotten a lot better!!)
python test.py --dataroot ./datasets/DWI_new --name DWI_new_pix2pix_grad_resnet9 --model pix2pix --direction BtoA --num_test 6660 --netG resnet_9blocks --batch_size 30
mv ./results/DWI_pix2pix_grad_resnet9_BtoA/test_latest ./results/DWI_new_pix2pix_grad_resnet9/Epoch_115

# then test on real data
#python test.py --dataroot ./datasets/DWI_sub-tle005 --name DWI_pix2pix_grad_resnet9 --model pix2pix --direction BtoA --num_test 5650 --netG resnet_9blocks --batch_size 30
#mv ./results/DWI_pix2pix_grad_resnet9_BtoA/test_latest ./results/DWI_pix2pix_grad_resnet9/sub-tle005_Epoch_115_real