#!/bin/bash
#SBATCH --job-name=classifier_s2s
#SBATCH --partition=dgx_regular
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
source /raid/soft/miniconda/bin/activate
conda init bash
conda activate profis
python train_clf_s2s.py -c config_files/MLP_config.ini
python train_clf_s2s.py -c config_files/RF_config.ini
python train_clf_s2s.py -c config_files/SVC_config.ini
python train_clf_s2s.py -c config_files/XGB_config.ini