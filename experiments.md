# Experiments
For our data we did multiple experiments using different setting. All of the commands are described below.

## Ablation studies
We started with some common ablation studies on the pix2pix model.

### Loss functions and generator architectures
As the default MSE loss, used for the GAN loss, blurred the images in some initial tests, we implemented a gradient-based sharpness loss as well as using the LPIPS metric as a loss. Then we also tested the different architecture options for the generator:

```bash
./train_tests_generators_lossFunctions.sh
```

The results where then analyzed using the following script:

```bash
python create_boxplots_tables_figures_experiments.py --experiment generators_lossFunctions
```
