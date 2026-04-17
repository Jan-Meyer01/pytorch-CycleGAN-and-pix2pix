def create_latex_table_loss_functions(latex_df, METRICS, save_path):
    for metric in METRICS:
        if metric in ["MSE", "DISTS", "GMSD"]:
            latex_df[metric] = latex_df.apply(
                lambda row: f"{row[f'{metric}_mean']*1000:.3f} $\\pm$ {row[f'{metric}_std']*1000:.3f}",
                axis=1
            )
        elif metric in ["SSIM", "FSIM"]:
            latex_df[metric] = latex_df.apply(
                lambda row: f"{row[f'{metric}_mean']*100:.2f} $\\pm$ {row[f'{metric}_std']*100:.2f}",
                axis=1
            )    

    # Keep only formatted columns
    latex_df = latex_df[["run"] + METRICS]

    latex_table = latex_df.to_latex(index=False, escape=False)
    latex_table = latex_table.replace("\\toprule", "\\hline")
    latex_table = latex_table.replace("\\midrule", "\\hline")
    latex_table = latex_table.replace("\\bottomrule", "\\hline")

    with open(save_path, "w") as f:
        f.write(latex_table)