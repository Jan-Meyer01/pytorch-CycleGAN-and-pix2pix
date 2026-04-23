# Experiments
For our data we did multiple experiments using different setting. All of the commands are described below.

## Ablation studies
We started with some common ablation studies on the pix2pix model.

### Loss functions
Besides the default MSE loss for the GAN loss we implemented a gradient-based sharpness loss as well as the LPIPS metric. Then we also tested the different combinations of MSE+Sharpness and LPIPS+Sharpness:

```bash
./train_tests_lossFunctions.sh
```

The results where then analyzed using the following script:

```bash
python create_boxplots_tables_figures_experiments.py --experiment lossFunctions
```

### Generator architectures
Next we used the best performing loss function, the sharpness loss, and tested different generator architectures:

```bash
./train_tests_generator.sh
```

For analysis use the same script as before, but for the generator experiment:

```bash
python create_boxplots_tables_figures_experiments.py --experiment generator
```