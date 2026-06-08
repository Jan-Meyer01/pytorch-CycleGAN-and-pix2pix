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
from util.latex_table import create_latex_table_overview
from util.figures     import create_overview_figure, create_overview_diff_figure

parser = argparse.ArgumentParser(description="Create boxplots, LaTeX tables, and example figures for different experiments.")
parser.add_argument("--base_dir",   type=str,  default="./results",                              help="Base directory containing the run subdirectories.")
parser.add_argument("--csv_name",   type=str,  default="metrics.csv",                            help="Name of the CSV file containing metrics in each run's test_latest directory.")
parser.add_argument("--metrics",    nargs="+", default=["MSE", "SSIM", "DISTS", "FSIM", "GMSD"], help="List of metrics to process.")
parser.add_argument("--num_images", type=int,  default=4440,                                     help="Number of images (rows) in the CSV file before the summary statistics start.")
args = parser.parse_args()

num_images = args.num_images

run_names = ["Epoch_100", "Epoch_115"]
save_path = './evaluation/before+after_fine-tuning'

all_runs_data_BtoA = []
summary_rows_BtoA  = []

for run_name in run_names:
    csv_path = os.path.join(args.base_dir, 'DWI_pix2pix_grad_resnet9_BtoA', run_name, args.csv_name)

    if not os.path.isfile(csv_path):
        print(f"Warning: CSV file not found for '{run_name}' at path '{csv_path}' - Skipping!")
        continue

    df = pd.read_csv(csv_path)

    # --- Split data ---
    image_data   = df.iloc[:num_images]  # per-image metrics
    summary_data = df.iloc[num_images:]  # mean + std

    # Add run label
    image_data["run"] = run_name
    all_runs_data_BtoA.append(image_data)    

    # Extract mean/std (assuming order: mean row, std row)
    mean_row = summary_data.iloc[0]
    std_row  = summary_data.iloc[1]

    summary_entry = {"run": run_name}
    for metric in args.metrics:
        summary_entry[f"{metric}_mean"] = mean_row[metric]
        summary_entry[f"{metric}_std"]  = std_row[metric]

    summary_rows_BtoA.append(summary_entry)

all_BtoA = pd.concat(all_runs_data_BtoA, ignore_index=True)

# create boxplots for each metric
os.makedirs(save_path, exist_ok=True)

for metric in args.metrics:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=all_BtoA, x="run", y=metric)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{metric}_boxplot_BtoA.png")
    plt.close()

# create LaTeX tables for summary statistics
create_latex_table_overview(pd.DataFrame(summary_rows_BtoA), args.metrics, f"{save_path}/pre-train_fine-tune_table.tex")

# create a new overview figure
create_overview_diff_figure([name for name in run_names], os.path.join(args.base_dir, 'DWI_pix2pix_grad_resnet9_BtoA'), 6, 6, f"{save_path}/overview_BtoA")
