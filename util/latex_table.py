from util.util import shorten_run_name

def create_latex_table_loss_functions(data_frame, METRICS, save_path):
    METRIC_DIRECTION = {
        "MSE": "down",
        "SSIM": "up",
        "DISTS": "down",
        "FSIM": "up",
        "GMSD": "down"
    }
    
    # Shorten run names
    data_frame["run"] = data_frame["run"].apply(shorten_run_name)

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
                value_str = f"{mean*1000:.3f} $\\pm$ {std*1000:.3f}"
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
   