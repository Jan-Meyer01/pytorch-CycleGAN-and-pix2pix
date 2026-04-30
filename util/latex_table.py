from util.util import parse_run_name, shorten_run_names_lossFunctions, shorten_run_names_generator
import pandas as pd

def create_latex_table_lossFunctions(data_frame, METRICS, save_path):    
    # Shorten run names for loss functions
    data_frame["run"] = data_frame["run"].apply(shorten_run_names_lossFunctions)

    # create table
    create_latex_table(data_frame, METRICS, save_path)
   
def create_latex_table_generator(data_frame, METRICS, save_path):
    # Shorten run names for generators
    data_frame["run"] = data_frame["run"].apply(shorten_run_names_generator)

    # create table
    create_latex_table(data_frame, METRICS, save_path)

def create_latex_table_generators_lossFunctions(data_frame, METRICS, save_path):
    METRIC_DIRECTION = {
        "MSE": "down",
        "SSIM": "up",
        "DISTS": "down",
        "FSIM": "up",
        "GMSD": "down"
    }

    # Parse run names
    data_frame["loss"], data_frame["architecture"] = zip(*data_frame["run"].apply(parse_run_name))
    
    # sort everything 
    loss_order = ["grad", "lpips"]
    arch_order = ["resnet6", "resnet9", "unet128", "unet256"]

    data_frame["loss"] = pd.Categorical(data_frame["loss"], loss_order)
    data_frame["architecture"] = pd.Categorical(data_frame["architecture"], arch_order)

    data_frame = data_frame.sort_values(["loss", "architecture"])
    
    # Determine best values (based on mean)
    best_indices = {}
    for metric in METRICS:
        if METRIC_DIRECTION[metric] == "down":
            best_idx = data_frame[f"{metric}_mean"].idxmin()
        else:
            best_idx = data_frame[f"{metric}_mean"].idxmax()
        best_indices[metric] = best_idx

    # Format values
    for metric in METRICS:
        formatted_col = []

        for idx, row in data_frame.iterrows():
            mean = row[f"{metric}_mean"]
            std  = row[f"{metric}_std"]

            if metric in ["MSE", "DISTS", "GMSD"]:
                value_str = f"{mean*1000:.2f} $\\pm$ {std*1000:.2f}"
            else:
                value_str = f"{mean*100:.2f} $\\pm$ {std*100:.2f}"

            # Bold best value
            if idx == best_indices[metric]:
                value_str = f"\\textbf{{{value_str}}}"

            formatted_col.append(value_str)

        data_frame[metric] = formatted_col

    # Add arrows to column names
    column_names = []
    for metric in METRICS:
        arrow = "$\\downarrow$" if METRIC_DIRECTION[metric] == "down" else "$\\uparrow$"
        column_names.append(f"{metric} {arrow}")

    # create columns for the table
    data_frame = data_frame[["loss", "architecture"] + METRICS]
    data_frame.columns = ["Loss", "Generator"] + column_names

    # create rows for the table
    rows = []
    current_loss = None

    for _, row in data_frame.iterrows():
        loss = shorten_run_names_lossFunctions(row["Loss"])

        if loss != current_loss:
            loss_str = f"\\multirow{{4}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{loss}}}}}"
            current_loss = loss
        else:
            loss_str = ""

        row_values = [loss_str, shorten_run_names_generator(row["Generator"])] + [row[m] for m in column_names]
        rows.append(" & ".join(row_values) + " \\\\")
    
    # create header for the table
    header = " & ".join(["Loss", "Generator"] + column_names) + " \\\\ \\hline"

    # create the final table
    latex_table = (
        "\\begin{tabular}{l l " + "c" * len(METRICS) + "}\n"
        "\\hline\n"
        + header + "\n"
        + "\\hline\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}"
    )

    # Save to file
    with open(save_path, "w") as f:
        f.write(latex_table)

def create_latex_table(data_frame, METRICS, save_path):
    METRIC_DIRECTION = {
        "MSE": "down",
        "SSIM": "up",
        "DISTS": "down",
        "FSIM": "up",
        "GMSD": "down"
    }
    
    # Determine best values (based on mean)
    best_indices = {}
    for metric in METRICS:
        if METRIC_DIRECTION[metric] == "down":
            best_idx = data_frame[f"{metric}_mean"].idxmin()
        else:
            best_idx = data_frame[f"{metric}_mean"].idxmax()
        best_indices[metric] = best_idx

    # Format values
    for metric in METRICS:
        formatted_col = []

        for idx, row in data_frame.iterrows():
            mean = row[f"{metric}_mean"]
            std  = row[f"{metric}_std"]

            if metric in ["MSE", "DISTS", "GMSD"]:
                value_str = f"{mean*1000:.2f} $\\pm$ {std*1000:.2f}"
            else:
                value_str = f"{mean*100:.2f} $\\pm$ {std*100:.2f}"

            # Bold best value
            if idx == best_indices[metric]:
                value_str = f"\\textbf{{{value_str}}}"

            formatted_col.append(value_str)

        data_frame[metric] = formatted_col

    # Add arrows to column names
    column_names = []
    for metric in METRICS:
        arrow = "$\\downarrow$" if METRIC_DIRECTION[metric] == "down" else "$\\uparrow$"
        column_names.append(f"{metric} {arrow}")

    # Keep only needed columns
    data_frame = data_frame[["run"] + METRICS]

    # Rename columns
    data_frame.columns = ["Loss"] + column_names

    # Prepare LaTeX table with center alignment and a thin line after the first column
    latex_table = data_frame.to_latex(index=False, escape=False, column_format="l" + "c" * len(METRICS)) #"|l|" + "c|"

    # Add thin line between the first and second columns
    #latex_table = latex_table.replace("\\begin{tabular}{|l", "\\begin{tabular}{|l|")
    
    # Replace rules
    latex_table = latex_table.replace("\\toprule", "\\hline")
    latex_table = latex_table.replace("\\midrule", "\\hline")
    latex_table = latex_table.replace("\\bottomrule", "\\hline")

    # Save to file
    with open(save_path, "w") as f:
        f.write(latex_table)
   