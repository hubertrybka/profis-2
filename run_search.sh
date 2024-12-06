#!/bin/bash
#SBATCH --job-name=search
#SBATCH --partition=dgx_A100
#SBATCH --cpus-per-task=256
#SBATCH --mem-per-cpu=16G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python bayesian_search.py -c config_files/search_config.ini