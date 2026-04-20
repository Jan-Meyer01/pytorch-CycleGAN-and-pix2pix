"""
Script to generate boxplots and tables for different loss functions.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # ignore warnings for cleaner output

rc('font', **{'family':'serif', 'serif': ['cmr10']})    # change plot font to Computer Modern Roman (used in LaTeX)

## custom imports
from util.latex_table import create_latex_table_loss_functions

# init values 
BASE_DIR   = "./results"
RUN_NAMES  = ["EPI_modelling_pix2pix_AtoB", "EPI_modelling_pix2pix_BtoA", 
              "EPI_modelling_pix2pix_lpips_AtoB", "EPI_modelling_pix2pix_lpips_BtoA", 
              "EPI_modelling_pix2pix_sharp_AtoB", "EPI_modelling_pix2pix_sharp_BtoA",
              "EPI_modelling_pix2pix_mseSharp_AtoB", "EPI_modelling_pix2pix_mseSharp_BtoA",
              "EPI_modelling_pix2pix_lpipsSharp_AtoB", "EPI_modelling_pix2pix_lpipsSharp_BtoA"]
CSV_NAME   = "metrics.csv"
METRICS    = ["MSE", "SSIM", "DISTS", "FSIM", "GMSD"]
NUM_IMAGES = 120

all_runs_data_AtoB = []
summary_rows_AtoB  = []

all_runs_data_BtoA = []
summary_rows_BtoA  = []

for run_name in RUN_NAMES:
    run_path = os.path.join(BASE_DIR, run_name)
    csv_path = os.path.join(run_path, 'test_latest', CSV_NAME)

    if not os.path.isfile(csv_path):
        print(f"Warning: CSV file not found for '{run_name}' at path '{csv_path}' - Skipping!")
        continue

    df = pd.read_csv(csv_path)

    # --- Split data ---
    image_data   = df.iloc[:NUM_IMAGES]
    summary_data = df.iloc[NUM_IMAGES:]  # mean + std

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
    for metric in METRICS:
        summary_entry[f"{metric}_mean"] = mean_row[metric]
        summary_entry[f"{metric}_std"]  = std_row[metric]

    if run_name.endswith("AtoB"):
        summary_rows_AtoB.append(summary_entry)
    elif run_name.endswith("BtoA"):
        summary_rows_BtoA.append(summary_entry)

all_AtoB        = pd.concat(all_runs_data_AtoB, ignore_index=True)
all_BtoA        = pd.concat(all_runs_data_BtoA, ignore_index=True)
summary_df_AtoB = pd.DataFrame(summary_rows_AtoB)
summary_df_BtoA = pd.DataFrame(summary_rows_BtoA)

# create boxplots for each metric
figure_path = './figures'
os.makedirs(figure_path, exist_ok=True)

for metric in METRICS:
    # plot AtoB
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=all_AtoB, x="run", y=metric)
    plt.tight_layout()
    plt.savefig(f"{figure_path}/{metric}_boxplot_AtoB.png")
    plt.close()

    # plot BtoA
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=all_BtoA, x="run", y=metric)
    plt.tight_layout()
    plt.savefig(f"{figure_path}/{metric}_boxplot_BtoA.png")
    plt.close()


# create LaTeX tables for summary statistics
create_latex_table_loss_functions(summary_df_AtoB.copy(), METRICS, "./results/results_table_lossFunctions_AtoB.tex")
create_latex_table_loss_functions(summary_df_BtoA.copy(), METRICS, "./results/results_table_lossFunctions_BtoA.tex")
