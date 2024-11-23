#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train.py --epochs 500 --batch_size 512 --lr 0.0002 --fp_type ECFP --name profis2_ecfp_lr2e-4