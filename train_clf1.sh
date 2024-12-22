#!/bin/bash
#SBATCH --job-name=classifier1
#SBATCH --partition=dgx_regular
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_clf_s2s.py -c config_files/MLP_config.ini