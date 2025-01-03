#!/bin/bash
#SBATCH --job-name=smiles2smiles
#SBATCH --partition=dgx_A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_SMILES2SMILES.py --epochs 600 --batch_size 512 --lr 0.0002 --name smiles2smiles_64 --eps_coef 1 --latent_size 64