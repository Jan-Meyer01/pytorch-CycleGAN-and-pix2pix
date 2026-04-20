import matplotlib.pyplot as plt
import numpy as np
import os

from util.util import shorten_run_name

def create_example_images(name, image_num):
    # read test examples
    results_dir = os.path.join('./results', name, 'test_latest', 'images')
    fake_img, real_img = find_real_fake_image(results_dir, image_num, mode='fake+real')

    # visualize the real and fake images
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes[0].imshow(fake_img[20:-20,20:-20])
    axes[0].set_title('Network Prediction', fontsize=20, loc='center', y=0.9, color='white')
    axes[0].axis('off')
    axes[1].imshow(real_img[20:-20,20:-20])
    axes[1].set_title('Target Image', fontsize=20, loc='center', y=0.9, color='white')
    axes[1].axis('off')
    plt.subplots_adjust(hspace=0.03)
    plt.savefig(os.path.join('./results', name, 'test_latest', name+'_test-comparison_{}.png'.format(image_num)), bbox_inches='tight')

def create_overview_figure(run_names, image_num):
    # split run names into AtoB and BtoA
    run_names_AtoB = [name for name in run_names if name.endswith("AtoB")]
    run_names_BtoA = [name for name in run_names if name.endswith("BtoA")]

    # create a new figure for each direction
    create_figure_runs(run_names_AtoB, image_num, "AtoB")
    create_figure_runs(run_names_BtoA, image_num, "BtoA")    

def create_figure_runs(run_names, image_num, direction):
    num_cols = int(np.ceil(len(run_names)/2))
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8.8))

    # read test examples for different runs
    for i, name in enumerate(run_names):
        results_dir = os.path.join('./results', name, 'test_latest', 'images')
        if i == 0:
            fake_img, real_img, input_img = find_real_fake_image(results_dir, image_num, mode='input+fake+real')
            # visualize the real and fake images
            axes[0,0].imshow(real_img[0:-20,20:-20])
            axes[0,0].set_title('Target Image', fontsize=20, loc='center', y=0.9, color='white')
            axes[0,0].axis('off')
            #axes[1,0].imshow(input_img[20:-20,20:-20])
            #axes[1,0].set_title('Input Image', fontsize=20, loc='center', y=0.9, color='white')
            #axes[1,0].axis('off')

        # visualize fake images for loss function comparison
        if i < len(run_names) // 2:
            axes[0,(i+1) % num_cols].imshow(fake_img[0:-20,20:-20])
            axes[0,(i+1) % num_cols].set_title(f'{shorten_run_name(name)}', fontsize=20, loc='center', y=0.9, color='white')
            axes[0,(i+1) % num_cols].axis('off')
        else:
            axes[1,i % num_cols].imshow(fake_img[0:-20,20:-20])
            axes[1,i % num_cols].set_title(f'{shorten_run_name(name)}', fontsize=20, loc='center', y=0.9, color='white')
            axes[1,i % num_cols].axis('off')

    # add title for the entire figure and save it
    if direction == "AtoB":
        fig.suptitle('Adding Artifacts', fontsize=24, y=0.92)
    else:
        fig.suptitle('Removing Artifacts', fontsize=24, y=0.92)

    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(os.path.join('./results', 'loss-comparison_{}_exampleImg{}.png'.format(direction, image_num)), dpi=300, bbox_inches='tight',pad_inches=0)

def find_real_fake_image(results_dir, image_num, mode='fake'):
    assert mode == 'fake' or mode == 'fake+real' or mode == 'input+fake+real', "Invalid mode. Must be 'fake', 'fake+real' or 'input+fake+real'."

    files = [f for f in os.listdir(results_dir) if f.endswith('.png') and f.find('_' + str(image_num)) != -1]
    fake_file = [f for f in files if f.find('fake') != -1][0]
    fake_img = plt.imread(os.path.join(results_dir, fake_file))
    if mode == 'fake':
        return fake_img
    elif mode == 'fake+real':
        real_file = fake_file.replace('fake', 'real')
        real_img = plt.imread(os.path.join(results_dir, real_file))
        return fake_img, real_img
    elif mode == 'input+fake+real':
        real_file = fake_file.replace('fake', 'real')
        real_img = plt.imread(os.path.join(results_dir, real_file))
        if fake_file.find('fake_A') != -1:
            input_file = fake_file.replace('fake_A', 'real_B')
        elif fake_file.find('fake_B') != -1:
            input_file = fake_file.replace('fake_B', 'real_A')
        input_img = plt.imread(os.path.join(results_dir, input_file))
        return fake_img, real_img, input_img