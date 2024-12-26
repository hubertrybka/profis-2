#!/bin/bash
#SBATCH --job-name=sample_posterior
#SBATCH --partition=dgx_A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python sample_from_posterior.py > sample_posterior.log