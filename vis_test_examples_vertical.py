import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name',        type=str, default='EPI_modelling_GE_pix2pix_BtoA', help='name of the experiment')
parser.add_argument('--model',       type=str, default='pix2pix',                       help='model type (pix2pix or cycle_gan)')
parser.add_argument('--num',         type=int, default=100,                             help='number of the test image to visualize')
opt = parser.parse_args()

# read test examples
results_dir = os.path.join('./results', opt.name, 'test_latest', 'images')
files = [f for f in os.listdir(results_dir) if f.endswith('.png') and f.find('_' + str(opt.num)) != -1]

# find fake image
fake_file = [f for f in files if f.find('fake') != -1][0]
fake_img = plt.imread(os.path.join(results_dir, fake_file))

# get the corresponding real image
real_file = fake_file.replace('fake', 'real')
real_img = plt.imread(os.path.join(results_dir, real_file))

# visualize the real and fake images
fig, axes = plt.subplots(2, 1, figsize=(5, 10))
axes[0].imshow(fake_img[20:-20,20:-20])
axes[0].set_title('Fake Image', fontsize=20, loc='center', y=0.9, color='white')
axes[0].axis('off')
axes[1].imshow(real_img[20:-20,20:-20])
axes[1].set_title('Real Image', fontsize=20, loc='center', y=0.9, color='white')
axes[1].axis('off')
plt.subplots_adjust(hspace=0.03)
plt.savefig(os.path.join('./results', opt.name, 'test_latest', 'comparison_{}.png'.format(opt.num)), bbox_inches='tight')