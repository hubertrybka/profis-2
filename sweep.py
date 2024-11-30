import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import torch.optim as optim
from rdkit import Chem
import wandb
from tqdm import tqdm
import os
import time
from profis.utils import decode_smiles_from_indexes, ProfisDataset, Annealer, load_charset
from profis.net import Profis, vae_loss

def is_valid(smiles):
    if Chem.MolFromSmiles(smiles, sanitize=True) is None:
        return False
    else:
        return True

def train(model, train_loader, val_loader, epochs=100, device='cuda', lr=0.0001, print_progress=False):

    charset = load_charset()
    annealer = Annealer(30, 'cosine', baseline=0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Using device:', device)

    for epoch in range(1, epochs + 1):

        print(f'Epoch', epoch)
        start_time = time.time()
        model.train()
        train_loss = 0
        mean_kld_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader)) if print_progress else enumerate(train_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, mean, logvar = model(X)
            recon_loss, kld_loss = vae_loss(output, y, mean, logvar)
            loss = recon_loss + annealer(kld_loss)
            loss.backward()
            train_loss += loss.item()
            kld_loss += kld_loss.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_loss /= len(train_loader)
        mean_kld_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_outputs = []
        for batch_idx, data in enumerate(val_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output, mean, logvar = model(X)
            if batch_idx % 100 == 0:
                print('Input:', decode_smiles_from_indexes(
                    y[0].argmax(dim=1).cpu().numpy(), charset).replace('[nop]', ''))
                print('Output:', decode_smiles_from_indexes(
                    output[0].argmax(dim=1).cpu().numpy(), charset).replace('[nop]', ''))
            loss, _ = vae_loss(output, y, mean, logvar)
            val_loss += loss.item()
            val_outputs.append(output.detach().cpu())
        val_loss /= len(val_loader)
        val_outputs = torch.cat(val_outputs, dim=0).numpy()
        val_out_smiles = [decode_smiles_from_indexes(
            out.argmax(axis=1), charset) for out in val_outputs]
        val_out_smiles = [smile.replace('[nop]', '') for smile in val_out_smiles]
        valid_smiles = [smile for smile in val_out_smiles if is_valid(smile)]
        mean_valid = len(valid_smiles) / len(val_out_smiles)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'validity': mean_valid})
        end_time = time.time()
        print(f'Epoch {epoch} completed in {(end_time - start_time)/60} min')

        #if epoch % 50 == 0:
        #    torch.save(model.state_dict(), f'models/{name}/epoch_{epoch}.pt')

    return model

def main():
    train_df = pd.read_parquet('data/RNN_dataset_ECFP_train_90.parquet')
    test_df = pd.read_parquet('data/RNN_dataset_ECFP_val_10.parquet')

    wandb.init(project="profis2-sweep")

    data_train = ProfisDataset(train_df, fp_len=2048)
    data_val = ProfisDataset(test_df, fp_len=2048)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=512, shuffle=False, num_workers=4)

    torch.manual_seed(42)

    if os.path.exists('models') is False:
        os.makedirs('models')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Profis(fp_size=2048,
                   dropout=wandb.config.dropout,
                   fc1_size=wandb.config.fc1_size,
                   fc2_size=wandb.config.fc2_size,
                   hidden_size=wandb.config.hidden_size,
                   eps_coef=wandb.config.eps_coef
                   ).to(device)
    model = train(model, train_loader, val_loader, epochs=500, device='cuda', lr=0.0002, print_progress=True)

sweep_config = {
    "method": "bayes",
    "name": "profis_sweep",
    "metric": {"goal": "maximize", "name": "validity"},
    "parameters": {
        "dropout": {"values": [0, 0.1, 0.2, 0.3]},
        "fc1_size": {"distribution": "int_uniform", "min": 256, "max": 2048},
        "fc2_size": {"distribution": "int_uniform", "min": 256, "max": 2048},
        "hidden_size": {"distribution": "int_uniform", "min": 256, "max": 1024},
        "eps_coef": {"values": [1, 0.1, 0.01]},
    },
    "early_terminate": {
        "type": "hyperband",
        "eta": 1.2,
        "min_iter": 100
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="profis2-sweep")
wandb.agent(sweep_id, function=main, count=100)