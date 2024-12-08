#!/bin/bash
#SBATCH --job-name=search
#SBATCH --partition=dgx_regular
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=16G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python bayesian_search.py -c config_files/search_config_1.ini > search_1.log
python bayesian_search.py -c config_files/search_config_2.ini > search_2.log