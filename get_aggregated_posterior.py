from profis.dataset import ProfisDataset, Smiles2SmilesDataset
import torch.utils.data as data
import json
from profis.net import MolecularVAE
from profis.utils import initialize_profis
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import configparser
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', '-m', type=str, required=True, help='Path to the model .pt file')
args = argparser.parse_args()

model_dir_path = '/'.join(args.model.split('/')[:-1])
config_path = model_dir_path + '/config.ini'

config = configparser.ConfigParser()
config.read(config_path)
encoding = config['MODEL']['in_encoding']

# determine aggregated posterior distribution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of samples to use for the calculation
n_sample = 100000

# random state for the sampling
rs = 42

if encoding in ['ECFP4', 'KRFP']:
    model = initialize_profis(config_path).to(device)
    model.load_state_dict(torch.load(args.model))
    df = pd.read_parquet(f"data/RNN_dataset_{encoding.strip('4')}_train_90.parquet").sample(n_sample, random_state=rs)
    train_dataset = ProfisDataset(df, fp_len=int(config['MODEL']['fp_len']))

elif encoding == 'SMILES':
    model = MolecularVAE().to(device)
    model.load_state_dict(torch.load(args.model))
    df = pd.read_parquet('data/RNN_dataset_ECFP_train_90.parquet').sample(n_sample, random_state=rs)
    train_dataset = Smiles2SmilesDataset(df)
else:
    raise ValueError("Invalid encoding format (must be 'ECFP4', 'KRFP', or 'SMILES')")

train_loader = data.DataLoader(train_dataset, batch_size=512, shuffle=False)

model.eval()
with torch.no_grad():
    posterior = []
    for batch_idx, data in enumerate(tqdm(train_loader)):
        if encoding == 'SMILES':
            X = data.to(device)
        else:
            X = data[0].to(device)
        output, mean, logvar = model(X)
        posterior.append(mean.detach().cpu().numpy())
    posterior = np.concatenate(posterior, axis=0)

# calculate aggregated posterior distribution
mean_posterior = posterior.mean(axis=0)
std_posterior = posterior.std(axis=0)

# save aggregated posterior distribution as json

with open(f'{model_dir_path}/aggregated_posterior.json', 'w') as f:
    save_dict = {'mean': mean_posterior.tolist(), 'std': std_posterior.tolist()}
    json.dump(save_dict, f)

