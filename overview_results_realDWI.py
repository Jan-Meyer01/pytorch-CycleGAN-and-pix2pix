import os
from matplotlib import rc
import argparse
import warnings
warnings.filterwarnings("ignore")  # ignore warnings for cleaner output

rc('font', **{'family':'serif', 'serif': ['cmr10']})    # change plot font to Computer Modern Roman (used in LaTeX)

## custom imports
from util.figures import create_comparison_four_b_values_figure

parser = argparse.ArgumentParser(description="Create boxplots, LaTeX tables, and example figures for different experiments.")
parser.add_argument("--base_dir",  type=str, default="./results",                help="Base directory containing the run subdirectories.")
parser.add_argument("--name",      type=str, default="DWI_pix2pix_grad_resnet9", help="Name of the model.")
parser.add_argument("--save_path", type=str, default="./evaluation/real_tests",  help="Directory for saving the figures.")
args = parser.parse_args()

base_dir  = args.base_dir
name      = args.name
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

# create a new overview figure (one per b-value)
subject = 'sub-tle005'
model_dir = os.path.join(base_dir, name, subject+'_Epoch_115_real')
image_num = 39
create_comparison_four_b_values_figure(model_dir, [0, 9, 3, 2,], image_num, f"{save_path}/{name}/{subject}")
