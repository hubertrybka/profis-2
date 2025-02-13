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
    disable_annealing=False,
    beta=1.0
):
    charset = load_charset()
    annealer = Annealer(1000, "cosine", baseline=0.0)
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
        annealed_kld_loss = 0
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
            mean_kld_loss += kld_loss.item()
            annealed_kld_loss += annealer(kld_loss).item()
            optimizer.step()
        train_loss /= len(train_loader)
        mean_kld_loss /= len(train_loader)
        mean_recon_loss /= len(train_loader)
        annealed_kld_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_outputs = []
        for batch_idx, data in enumerate(val_loader):
            X = data.to(device)
            output, mean, logvar = model(X)
            recon_loss, kld_loss = criterion(output, X, mean, logvar)
            loss = recon_loss + annealer(kld_loss)
            val_loss += loss.item()
            val_outputs.append(output.detach().cpu())
        val_loss /= len(val_loader)

        # Decode example SMILES
        output_smiles = decode_seq_from_output(val_outputs, charset)
        valid_seqs, mean_valid = validate_seqs(output_smiles, is_valid)

        # Try to sample from the latent space and decode
        latent_space = torch.randn(10000, 32).to(device)
        output = model.decode(latent_space)
        output_smiles = decode_seq_from_output(output, charset)
        sampled_seqs, sampled_valid = validate_seqs(output_smiles, is_valid)

        annealer.step()
        wandb.log(
            {"train_loss": train_loss,
             "val_loss": val_loss,
             "validity": mean_valid,
             "kld_loss_train": mean_kld_loss,
             "recon_loss_train": mean_recon_loss,
             "annealed_kld_loss": annealed_kld_loss,
             "output_smiles": output_smiles[:16],
             "sampling_validity": sampled_valid,
             "sampled_seqs": sampled_seqs[:16]
             }
        )

        None if disable_annealing else annealer.step()

        end_time = time.time()
        print(f"Epoch {epoch} completed in {(end_time - start_time)/60} min")

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"models/{args.name}/epoch_{epoch}.pt")

    return model

def decode_seq_from_output(output: list[torch.Tensor], charset: list[str]) -> list[str]:
    print("Decoding sequences from output")
    print("Output type:", type(output))
    print("Output[0] shape:", output[0].shape)

    output = torch.cat(output, dim=0,).numpy()
    out_seq = [
        decode_seq_from_indexes(out.argmax(axis=1), charset) for out in output
    ]
    # Remove the [nop] tokens
    output_smiles = [seq.replace("[nop]", "") for seq in out_seq]
    return output_smiles

def validate_seqs(seq_list, is_valid):
    valid_seqs = [seq for seq in seq_list if is_valid(seq)]
    validity = len(valid_seqs) / len(seq_list)
    return valid_seqs, validity

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
argparser.add_argument("--disable_annealing", action="store_true", default=False)
argparser.add_argument("--beta", type=float, default=1.0)
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
    print_progress=False,
    disable_annealing=args.disable_annealing,
    beta=float(args.beta)
)
