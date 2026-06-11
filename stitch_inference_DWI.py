import os
from os.path import join
import argparse
import matplotlib.pyplot as plt
import numpy as np
import ants

parser = argparse.ArgumentParser(description="Create boxplots, LaTeX tables, and example figures for different experiments.")
parser.add_argument("--base_dir",  type=str, default="./results",                     help="Base directory containing the run subdirectories.")
parser.add_argument("--name",      type=str, default="DWI_pix2pix_grad_resnet9",      help="Name of the model.")
parser.add_argument("--save_path", type=str, default="./evaluation/real_tests/data",  help="Directory for saving the figures.")
args = parser.parse_args()

base_dir  = args.base_dir
name      = args.name
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
subject   = 'sub-tle005_'
image_dir = join(base_dir, name, subject+'Epoch_115_real','images')

# get orientation, etc. for later
ref_image = ants.image_read('/projects/crunchie/Jan/Daten/DataTLE/Raw/invivo/sub-tle005/ses-preop/additional/preprocessed/p005_SMI_merged_stripped.nii')

# find network inference "fake" images
fake_files   = [f for f in os.listdir(image_dir) if f.endswith('.png') and f.find('fake') != -1]
fake_images  = np.zeros((256,256,50,113))
real_images  = np.zeros((256,256,50,113))
input_images = np.zeros((256,256,50,113))

for fake_file in fake_files:
    # split file name and get slice number and diffusion direction number
    fake_file_split = fake_file.split('_')
    diff_dir_num = int(fake_file_split[1])
    slice_num    = int(fake_file_split[2])

    # make image grayscale and save it for later
    img = np.mean(plt.imread(join(image_dir, fake_file)),axis=-1)
    fake_images[:,:,slice_num,diff_dir_num] = img

    # exchange fake for real and also save it
    real_file = fake_file.replace('fake', 'real')
    img = np.mean(plt.imread(join(image_dir, real_file)),axis=-1)
    real_images[:,:,slice_num,diff_dir_num] = img

    # also save the input image for completeness
    if real_file.find('real_A') != -1:
        img = np.mean(plt.imread(join(image_dir, real_file.replace('real_A', 'real_B'))),axis=-1)
    else:
        img = np.mean(plt.imread(join(image_dir, real_file.replace('real_B', 'real_A'))),axis=-1)
    input_images[:,:,slice_num,diff_dir_num] = img

# load reference values
ref_dir = './datasets/DWI_sub-tle005'
ref_val_unproc = np.load(join(ref_dir,'ref_val_unproc.npy'))
ref_val_proc   = np.load(join(ref_dir,'ref_val_proc.npy'))

# rescale all the images to what they were before the network conversion
for i in range(ref_val_unproc.shape[0]):
    # use values of the unprocessed base image for input and network output
    input_images[:,:,:,i] = input_images[:,:,:,i] * ref_val_unproc[i]
    fake_images[:,:,:,i]  = fake_images[:,:,:,i] * ref_val_unproc[i]
    
    # use values for processed image for the target image
    real_images[:,:,:,i] = real_images[:,:,:,i] * ref_val_proc[i]

# reorient them back using flip and transpose the first two axis
input_images = np.flip(input_images,axis=0)
fake_images  = np.flip(fake_images,axis=0)
real_images  = np.flip(real_images,axis=0)

input_images = np.permute_dims(input_images, axes=[1,0,2,3])
fake_images  = np.permute_dims(fake_images, axes=[1,0,2,3])
real_images  = np.permute_dims(real_images, axes=[1,0,2,3])

# save both as niftis
ants.image_write(ants.from_numpy(fake_images, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction),join(save_path,name,'fake_img.nii'))
ants.image_write(ants.from_numpy(real_images, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction),join(save_path,name,'real_img.nii'))
ants.image_write(ants.from_numpy(input_images, origin=ref_image.origin, spacing=ref_image.spacing, direction=ref_image.direction),join(save_path,name,'input_img.nii'))

# create brain masks for each image
ants.image_write(ants.from_numpy((input_images[:,:,:,0] > 0).astype(np.float32), origin=ref_image.origin[:3], spacing=ref_image.spacing[:3], direction=ref_image.direction[:3, :3]),join(save_path,name,'input_mask.nii'))
ants.image_write(ants.from_numpy((fake_images[:,:,:,0] > 0).astype(np.float32), origin=ref_image.origin[:3], spacing=ref_image.spacing[:3], direction=ref_image.direction[:3, :3]),join(save_path,name,'fake_mask.nii'))
ants.image_write(ants.from_numpy((real_images[:,:,:,0] > 0).astype(np.float32), origin=ref_image.origin[:3], spacing=ref_image.spacing[:3], direction=ref_image.direction[:3, :3]),join(save_path,name,'real_mask.nii'))