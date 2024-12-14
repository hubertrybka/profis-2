#!/bin/bash
#SBATCH --job-name=posterior
#SBATCH --partition=dgx_A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python get_aggregated_posterior.py -m models/eps1_dropout_ECFP/epoch_600.pt
python get_aggregated_posterior.py -m models/eps1_dropout_KRFP/epoch_600.pt