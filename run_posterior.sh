#!/bin/bash
#SBATCH --job-name=posterior
#SBATCH --partition=dgx_A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python get_aggregated_posterior.py -m models/ECFP_DeepSMILES/epoch_600.pt
python get_aggregated_posterior.py -m models/KRFP_DeepSMILES/epoch_600.pt
python get_aggregated_posterior.py -m models/ECFP_SELFIES/epoch_600.pt
python get_aggregated_posterior.py -m models/KRFP_SELFIES/epoch_600.pt