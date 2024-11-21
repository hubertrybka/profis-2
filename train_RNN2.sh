#!/bin/bash
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python train.py --epochs 300 --batch_size 512 --lr 0.0001 --fp_type KRFP --name profis2_krfp_lr2e-4