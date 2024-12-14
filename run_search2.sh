#!/bin/bash
#SBATCH --job-name=search2
#SBATCH --partition=dgx_regular
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=8G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python bayesian_search.py -c config_files/search_config_KRFP.ini