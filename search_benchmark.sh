#!/bin/bash
#SBATCH --job-name=search
#SBATCH --partition=dgx_regular
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python bayesian_search.py -c config_files/search_config_5.ini
python bayesian_search.py -c config_files/search_config_10.ini
python bayesian_search.py -c config_files/search_config_25.ini
python bayesian_search.py -c config_files/search_config_50.ini
python bayesian_search.py -c config_files/search_config_100.ini