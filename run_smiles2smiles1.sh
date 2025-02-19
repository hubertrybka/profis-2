#!/bin/bash
#SBATCH --job-name=smiles2smiles_2
#SBATCH --partition=dgx_A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_SMILES2SMILES.py --epochs 1000 --batch_size 512 --lr 0.0003 --name smiles2smiles_fixed2 --eps_coef 1 --latent_size 32 --beta 0.1