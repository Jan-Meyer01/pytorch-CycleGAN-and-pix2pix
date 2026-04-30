import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import argparse
import warnings
warnings.filterwarnings("ignore")  # ignore warnings for cleaner output

rc('font', **{'family':'serif', 'serif': ['cmr10']})    # change plot font to Computer Modern Roman (used in LaTeX)

## custom imports
from util.latex_table import create_latex_table_generators_lossFunctions
from util.figures     import create_figure_runs_generators_lossFunctions

parser = argparse.ArgumentParser(description="Create boxplots, LaTeX tables, and example figures for different experiments.")
parser.add_argument("--base_dir",   type=str,  default="./results",                              help="Base directory containing the run subdirectories.")
parser.add_argument("--experiment", type=str,  default="generators_lossFunctions",               help="Name of the experiment - either 'lossFunctions', 'generators' or 'generators_lossFunctions'.")
parser.add_argument("--csv_name",   type=str,  default="metrics.csv",                            help="Name of the CSV file containing metrics in each run's test_latest directory.")
parser.add_argument("--metrics",    nargs="+", default=["MSE", "SSIM", "DISTS", "FSIM", "GMSD"], help="List of metrics to process.")
parser.add_argument("--num_images", type=int,  default=120,                                      help="Number of images (rows) in the CSV file before the summary statistics start.")
args = parser.parse_args()

assert args.experiment in ["lossFunctions", "generators", "generators_lossFunctions"], "Experiment name must be either 'lossFunctions', 'generators' or 'generators_lossFunctions'."

num_images = args.num_images

if args.experiment == "generators_lossFunctions":
    run_names  = ["EPI_modelling_SE_pix2pix_grad_resnet6_AtoB",  "EPI_modelling_SE_pix2pix_grad_resnet6_BtoA",
                  "EPI_modelling_SE_pix2pix_grad_resnet9_AtoB",  "EPI_modelling_SE_pix2pix_grad_resnet9_BtoA",
                  "EPI_modelling_SE_pix2pix_grad_unet128_AtoB",  "EPI_modelling_SE_pix2pix_grad_unet128_BtoA",
                  "EPI_modelling_SE_pix2pix_grad_unet256_AtoB",  "EPI_modelling_SE_pix2pix_grad_unet256_BtoA",
                  "EPI_modelling_SE_pix2pix_lpips_resnet6_AtoB", "EPI_modelling_SE_pix2pix_lpips_resnet6_BtoA",
                  "EPI_modelling_SE_pix2pix_lpips_resnet9_AtoB", "EPI_modelling_SE_pix2pix_lpips_resnet9_BtoA",
                  "EPI_modelling_SE_pix2pix_lpips_unet128_AtoB", "EPI_modelling_SE_pix2pix_lpips_unet128_BtoA",
                  "EPI_modelling_SE_pix2pix_lpips_unet256_AtoB", "EPI_modelling_SE_pix2pix_lpips_unet256_BtoA"]
    num_images = 60         # only 60 images as we only use SE

save_path = './evaluation/{}'.format(args.experiment)

all_runs_data_AtoB = []
summary_rows_AtoB  = []

all_runs_data_BtoA = []
summary_rows_BtoA  = []

for run_name in run_names:
    run_path = os.path.join(args.base_dir, run_name)
    csv_path = os.path.join(run_path, 'test_latest', args.csv_name)

    if not os.path.isfile(csv_path):
        print(f"Warning: CSV file not found for '{run_name}' at path '{csv_path}' - Skipping!")
        continue

    df = pd.read_csv(csv_path)

    # --- Split data ---
    image_data   = df.iloc[:num_images]  # per-image metrics
    summary_data = df.iloc[num_images:]  # mean + std

    # Add run label
    image_data["run"] = run_name
    if run_name.endswith("AtoB"):
        all_runs_data_AtoB.append(image_data)
    elif run_name.endswith("BtoA"):
        all_runs_data_BtoA.append(image_data)    

    # Extract mean/std (assuming order: mean row, std row)
    mean_row = summary_data.iloc[0]
    std_row  = summary_data.iloc[1]

    summary_entry = {"run": run_name}
    for metric in args.metrics:
        summary_entry[f"{metric}_mean"] = mean_row[metric]
        summary_entry[f"{metric}_std"]  = std_row[metric]

    if run_name.endswith("AtoB"):
        summary_rows_AtoB.append(summary_entry)
    elif run_name.endswith("BtoA"):
        summary_rows_BtoA.append(summary_entry)

all_AtoB = pd.concat(all_runs_data_AtoB, ignore_index=True)
all_BtoA = pd.concat(all_runs_data_BtoA, ignore_index=True)

# create boxplots for each metric
os.makedirs(save_path, exist_ok=True)

for metric in args.metrics:
    # plot AtoB
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=all_AtoB, x="run", y=metric)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{args.experiment}_{metric}_boxplot_AtoB.png")
    plt.close()

    # plot BtoA
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=all_BtoA, x="run", y=metric)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{args.experiment}_{metric}_boxplot_BtoA.png")
    plt.close()

if args.experiment == "generators_lossFunctions":
    # create LaTeX tables for summary statistics
    create_latex_table_generators_lossFunctions(pd.DataFrame(summary_rows_AtoB), args.metrics, f"{save_path}/generators_lossFunctions_table_AtoB.tex")
    create_latex_table_generators_lossFunctions(pd.DataFrame(summary_rows_BtoA), args.metrics, f"{save_path}/generators_lossFunctions_table_BtoA.tex")

    # create a new figure for each direction
    create_figure_runs_generators_lossFunctions([name for name in run_names if name.endswith("AtoB")], 148, f"{save_path}/generators_lossFunctions_AtoB")
    create_figure_runs_generators_lossFunctions([name for name in run_names if name.endswith("BtoA")], 148, f"{save_path}/generators_lossFunctions_BtoA")
