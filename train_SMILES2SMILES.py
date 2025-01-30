import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import torch.optim as optim
from rdkit import Chem
import wandb
import argparse
from tqdm import tqdm
import os
import time
from profis.dataset import (
    load_charset,
    Smiles2SmilesDataset
)
from profis.utils import decode_seq_from_indexes
from profis.net import MolecularVAE, VaeLoss, Annealer


def is_valid(smiles):
    if Chem.MolFromSmiles(smiles, sanitize=True) is None:
        return False
    else:
        return True


def train(
    model,
    train_loader,
    val_loader,
    epochs=100,
    device="cuda",
    lr=0.0004,
    print_progress=False,
    disable_annealing=False
):

    beta = 0.01

    charset = load_charset()
    annealer = Annealer(30, "cosine", baseline=0.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Using device:", device)

    criterion = VaeLoss()

    for epoch in range(1, epochs + 1):

        print(f"Epoch", epoch)
        start_time = time.time()
        model.train()
        train_loss = 0
        mean_kld_loss = 0
        mean_recon_loss = 0
        for batch_idx, data in (
            enumerate(tqdm(train_loader)) if print_progress else enumerate(train_loader)
        ):
            X = data.to(device)
            optimizer.zero_grad()
            output, mean, logvar = model(X)
            recon_loss, kld_loss = criterion(output, X, mean, logvar)
            loss = recon_loss + beta * annealer(kld_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss += loss.item()
            mean_recon_loss += recon_loss.item()
            mean_kld_loss += beta * kld_loss.item()
            optimizer.step()
        train_loss /= len(train_loader)
        mean_kld_loss /= len(train_loader)
        mean_recon_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_outputs = []
        for batch_idx, data in enumerate(val_loader):
            X = data.to(device)
            output, mean, logvar = model(X)
            if batch_idx % 50 == 0:
                print(
                    "Input:",
                    decode_seq_from_indexes(
                        X[0].argmax(dim=1).cpu().numpy(), charset
                    ).replace("[nop]", ""),
                )
                print(
                    "Output:",
                    decode_seq_from_indexes(
                        output[0].argmax(dim=1).cpu().numpy(), charset
                    ).replace("[nop]", ""),
                )
            recon_loss, kld_loss = criterion(output, X, mean, logvar)
            loss = recon_loss + annealer(kld_loss)
            val_loss += loss.item()
            val_outputs.append(output.detach().cpu())
        val_loss /= len(val_loader)
        val_outputs = torch.cat(val_outputs, dim=0).numpy()
        val_out_smiles = [
            decode_seq_from_indexes(out.argmax(axis=1), charset)
            for out in val_outputs
        ]
        val_out_smiles = [smile.replace("[nop]", "") for smile in val_out_smiles]
        valid_smiles = [smile for smile in val_out_smiles if is_valid(smile)]
        mean_valid = len(valid_smiles) / len(val_out_smiles)

        wandb.log(
            {"train_loss": train_loss,
             "val_loss": val_loss,
             "validity": mean_valid,
             "annealing": annealer(1),
             "recon_loss_train": mean_recon_loss,
             "kld_loss_train": mean_kld_loss},
        )

        annealer.step() if disable_annealing is False else None

        end_time = time.time()
        print(f"Epoch {epoch} completed in {(end_time - start_time)/60} min")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"models/{args.name}/epoch_{epoch}.pt")

    return model

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--epochs", type=int, default=100, help="Number of epochs to train the model"
)
argparser.add_argument("--batch_size", type=int, default=128)
argparser.add_argument(
    "--lr", type=float, default=0.0001, help="Learning rate for the optimizer"
)
argparser.add_argument("--name", type=str, default="profis")
argparser.add_argument("--eps_coef", type=float, default=0.01)
argparser.add_argument("--dropout", type=float, default=0.2)
argparser.add_argument("--latent_size", type=int, default=32)
argparser.add_argument("--disable_annealing", action="store_true")
args = argparser.parse_args()

wandb.init(project="profis2", name=args.name, config=args)

train_df = pd.read_parquet("data/RNN_dataset_ECFP_train_90.parquet")
test_df = pd.read_parquet("data/RNN_dataset_ECFP_val_10.parquet")
data_train = Smiles2SmilesDataset(train_df)
data_val = Smiles2SmilesDataset(test_df)
train_loader = torch.utils.data.DataLoader(
    data_train, batch_size=args.batch_size, shuffle=True, num_workers=4
)
val_loader = torch.utils.data.DataLoader(
    data_val, batch_size=args.batch_size, shuffle=False, num_workers=4
)

torch.manual_seed(42)

if os.path.exists("models") is False:
    os.makedirs("models")

if os.path.exists(f"models/{args.name}") is False:
    os.makedirs(f"models/{args.name}")

epochs = args.epochs
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MolecularVAE(dropout=args.dropout, latent_size=args.latent_size).to(device)
model = train(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    lr=args.lr,
    print_progress=True,
    disable_annealing=args.disable_annealing,
)
