# Tests for different loss functions. 

# default MSE loss for GAN loss in pix2pix
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_BtoA --model pix2pix --direction BtoA --num_test 120

# LPIPS as a loss
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpips_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpips_BtoA --model pix2pix --direction BtoA --num_test 120

# sharpness (gradient) loss
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_sharp_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_sharp_BtoA --model pix2pix --direction BtoA --num_test 120

# MSE + sharpness loss
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_mseSharp_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_mseSharp_BtoA --model pix2pix --direction BtoA --num_test 120

# LPIPS + sharpness loss
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpipsSharp_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpipsSharp_BtoA --model pix2pix --direction BtoA --num_test 120