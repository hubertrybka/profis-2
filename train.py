import torch
import torch.utils.data
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import wandb
import configparser
from tqdm import tqdm
import argparse
import os
import time
from profis.utils import (
    load_charset,
    initialize_profis,
    ValidityChecker,
    decode_seq_from_indexes,
)
from profis.net import VaeLoss, Annealer
from profis.dataset import ProfisDataset, DeepSmilesDataset, SelfiesDataset


def train(
    model,
    train_loader,
    val_loader,
    annealer,
    epochs=100,
    device="cpu",
    lr=0.0001,
    beta=1.0,
    name="profis",
    out_encoding="smiles",
    print_progress=False,
):

    charset = load_charset(f"data/{out_encoding}_alphabet.txt")
    is_valid = ValidityChecker(out_encoding)

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

        # train loop

        for batch_idx, data in (
            enumerate(tqdm(train_loader)) if print_progress else enumerate(train_loader)
        ):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, mean, logvar = model(X)
            recon_loss, kld_loss = criterion(output, y, mean, logvar)
            loss = recon_loss + beta * annealer(kld_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_loss += loss.item()
            mean_kld_loss += kld_loss.item()
            annealed_kld_loss += annealer(kld_loss).item()
            mean_recon_loss += recon_loss.item()
            optimizer.step()
        train_loss /= len(train_loader)
        mean_kld_loss /= len(train_loader)
        mean_recon_loss /= len(train_loader)
        annealed_kld_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        val_outputs = []
        for batch_idx, data in enumerate(val_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            output, mean, logvar = model(X)
            loss, kld_loss = criterion(output, y, mean, logvar)
            loss = loss + annealer(kld_loss)
            val_loss += loss.item()
            val_outputs.append(output.detach().cpu())
        val_loss /= len(val_loader)

        # Decode example SMILES
        val_outputs = torch.cat(val_outputs, dim=0).numpy()
        output_smiles = decode_seq_from_output(val_outputs, charset)
        valid_seqs, mean_valid = validate_seqs(output_smiles, is_valid)

        # Try to sample from the latent space and decode
        latent_space = torch.randn(10000, 32).to(device)
        output = model.decode(latent_space)
        output_seqs = decode_seq_from_output(output, charset)
        sampled_seqs, sampled_validity = validate_seqs(output_seqs, is_valid)

        annealer.step()
        wandb.log(
            {"train_loss": train_loss,
             "val_loss": val_loss,
             "validity": mean_valid,
             "kld_loss_train": mean_kld_loss,
             "recon_loss_train": mean_recon_loss,
             "annealed_kld_loss": annealed_kld_loss,
             "output_smiles": output_smiles[:16],
             "sampling_validity": sampled_validity,
             "sampled_seqs": sampled_seqs[:16]
             }
        )
        end_time = time.time()
        print(f"Epoch {epoch} completed in {(end_time - start_time)/60} min")

        if epoch % 50 == 0:
            # save the model
            torch.save(model.state_dict(), f"models/{name}/epoch_{epoch}.pt")

    return model

def decode_seq_from_output(output, charset):
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

if __name__ == "__main__":

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    # Parse the path to the config file
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config", "-c", type=str, default="config_files/RNN_config.ini"
    )
    args = argparser.parse_args()

    # Read the config file
    parser = configparser.ConfigParser()
    parser.read(args.config)

    # Initialize wandb
    wandb.init(project="profis2", name=parser["RUN"]["run_name"], config=parser)

    # Load the data and create the dataloaders
    fp_type = parser["MODEL"]["in_encoding"]
    train_df = pd.read_parquet(
        "data/RNN_dataset_ECFP_train_90.parquet"
        if fp_type == "ECFP4"
        else "data/RNN_dataset_KRFP_train_90.parquet"
    )
    test_df = pd.read_parquet(
        "data/RNN_dataset_ECFP_val_10.parquet"
        if fp_type == "ECFP4"
        else "data/RNN_dataset_KRFP_val_10.parquet"
    )

    out_encoding = parser["RUN"]["out_encoding"]
    fp_len = int(parser["MODEL"]["fp_len"])
    overwrite_model_dir = parser["RUN"].getboolean("overwrite_model_dir")

    if out_encoding.lower() == "smiles":
        data_train = ProfisDataset(train_df, fp_len=fp_len)
        data_val = ProfisDataset(test_df, fp_len=fp_len)
    elif out_encoding.lower() == "deepsmiles":
        data_train = DeepSmilesDataset(train_df, fp_len=fp_len)
        data_val = DeepSmilesDataset(test_df, fp_len=fp_len)
    elif out_encoding.lower() == "selfies":
        data_train = SelfiesDataset(train_df, fp_len=fp_len)
        data_val = SelfiesDataset(test_df, fp_len=fp_len)

    else:
        raise ValueError(f"Invalid output encoding: {out_encoding}")

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=int(parser["RUN"]["batch_size"]),
        shuffle=True,
        num_workers=int(parser["RUN"]["num_workers"]),
    )
    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=int(parser["RUN"]["batch_size"]),
        shuffle=False,
        num_workers=int(parser["RUN"]["num_workers"]),
    )
    torch.manual_seed(42)

    # Create a directory to save the models to
    if os.path.exists("models") is False:
        os.makedirs("models")
    model_name = parser["RUN"]["run_name"]
    if os.path.exists(f"models/{model_name}") is False:
        os.makedirs(f"models/{model_name}")
    elif overwrite_model_dir:
        pass
    else:
        raise ValueError(f"Model directory of the name {model_name} already exists")

    # Dump the config file to the model directory
    with open(f"models/{model_name}/config.ini", "w") as f:
        parser.write(f)

    # Initialize the annealing agent
    annealer = Annealer(
        int(parser["RUN"]["annealing_max_epoch"]),
        parser["RUN"]["annealing_shape"],
        baseline=0.0,
        disable=parser["RUN"].getboolean("disable_annealing"),
    )

    # Initialize the model
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if parser["RUN"].getboolean("use_cuda")
        else "cpu"
    )
    model = initialize_profis(args.config)
    model.to(device)

    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(parser["RUN"]["epochs"]),
        annealer=annealer,
        device=device,
        lr=float(parser["RUN"]["learn_rate"]),
        name=model_name,
        beta=float(parser["RUN"]["beta"]),
        out_encoding=parser["RUN"]["out_encoding"],
        print_progress=False,
    )
