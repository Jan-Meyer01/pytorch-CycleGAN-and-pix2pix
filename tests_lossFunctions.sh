#!/bin/bash


python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_BtoA --model pix2pix --direction BtoA --num_test 120

python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpips_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpips_BtoA --model pix2pix --direction BtoA --num_test 120

python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpipsSharp_AtoB --model pix2pix --direction AtoB --num_test 120
python test.py --dataroot ./datasets/EPI_modelling --name EPI_modelling_pix2pix_lpipsSharp_BtoA --model pix2pix --direction BtoA --num_test 120