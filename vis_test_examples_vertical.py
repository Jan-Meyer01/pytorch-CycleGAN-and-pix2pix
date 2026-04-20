import argparse

from util.figures import create_example_images

parser = argparse.ArgumentParser()
parser.add_argument('--name',        type=str, default='EPI_modelling_GE_pix2pix_BtoA', help='name of the experiment')
parser.add_argument('--model',       type=str, default='pix2pix',                       help='model type (pix2pix or cycle_gan)')
parser.add_argument('--num',         type=int, default=100,                             help='number of the test image to visualize')
opt = parser.parse_args()

create_example_images(opt.name, opt.num)