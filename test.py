"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import csv
from piq import ssim, fsim, DISTS, gmsd

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    #print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    
    # init metrics
    mse_values   = []
    ssim_values  = []
    dists_values = []
    fsim_values  = []
    gmsd_values  = []
    #D = DISTS()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        #if i % 5 == 0:  # save images to an HTML file
        #    print(f"processing ({i:04d})-th image... {img_path}")
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
        # put images onto the CPU and go from [-1,1] to [0,1]
        real = visuals["real_B"].squeeze().cpu()#.numpy() #np.mean(, 0)
        fake = visuals["fake_B"].squeeze().cpu()#.numpy() #np.mean(, 0)
        real = (real+1)/2
        fake = (fake+1)/2

        # calculate image metrics
        mse_values.append(np.mean((real.numpy()[real > 0] - fake.numpy()[real > 0]) ** 2))
        ssim_values.append(ssim(real.unsqueeze(0), fake.unsqueeze(0), data_range=1.).detach().numpy())
        dists_values.append((DISTS()(real.unsqueeze(0), fake.unsqueeze(0))).detach().numpy())
        fsim_values.append(fsim(real.unsqueeze(0), fake.unsqueeze(0), data_range=1.).detach().numpy())
        gmsd_values.append(gmsd(real.unsqueeze(0), fake.unsqueeze(0), data_range=1.).detach().numpy())
    webpage.save()  # save the HTML
    
    # compute and print average metrics
    mean_mse    = np.mean(mse_values)
    std_mse     = np.std(mse_values)
    mean_ssim   = np.mean(ssim_values)
    std_ssim    = np.std(ssim_values)
    mean_dists  = np.mean(dists_values)
    std_dists   = np.std(dists_values)
    mean_fsim   = np.mean(fsim_values)
    std_fsim    = np.std(fsim_values)
    mean_gmsd   = np.mean(gmsd_values)
    std_gmsd    = np.std(gmsd_values)
    print("Average MSE in e-3:   {:.3f} +- {:.3f}".format(mean_mse * 1000, std_mse * 1000))
    print("Average SSIM in %:    {:.2f} +- {:.2f}".format(mean_ssim * 100, std_ssim * 100))
    print("Average DISTS in e-3: {:.2f} +- {:.2f}".format(mean_dists * 1000, std_dists * 1000))
    print("Average FSIM in %:    {:.2f} +- {:.2f}".format(mean_fsim * 100, std_fsim * 100))
    print("Average GMSD  in e-3: {:.2f} +- {:.2f}".format(mean_gmsd * 1000, std_gmsd * 1000))
    
    # save metrics to csv file
    csv_path = os.path.join(web_dir, "metrics.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['Image Number', 'MSE', 'SSIM', 'DISTS', 'FSIM', 'GMSD']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, (mse_value, ssim_value, dists_value, fsim_value, gmsd_value) in enumerate(zip(mse_values, ssim_values, dists_values, fsim_values, gmsd_values)):
            writer.writerow({'Image Number': i, 'MSE': mse_value, 'SSIM': ssim_value, 'DISTS': dists_value, 'FSIM': fsim_value, 'GMSD': gmsd_value})
        writer.writerow({'Image Number': 'Mean', 'MSE': mean_mse, 'SSIM': mean_ssim, 'DISTS': mean_dists, 'FSIM': mean_fsim, 'GMSD': mean_gmsd})
        writer.writerow({'Image Number': 'Std', 'MSE': std_mse, 'SSIM': std_ssim, 'DISTS': std_dists, 'FSIM': std_fsim, 'GMSD': std_gmsd})
