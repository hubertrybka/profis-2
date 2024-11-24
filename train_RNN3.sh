#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train_SMILES2SMILES.py --epochs 500 --batch_size 512 --lr 0.0002 --name smiles2smiles_lr2e-4