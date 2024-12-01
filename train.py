import torch
import torch.utils.data
import pandas as pd
import torch.optim as optim
import wandb
import configparser
from tqdm import tqdm
import argparse
import os
import time
from profis.utils import Annealer, load_charset, initialize_profis, is_valid
from profis.net import vae_loss
from profis.dataset import ProfisDataset

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def train(model,
          train_loader,
          val_loader,
          annealer,
          epochs=100,
          device='cpu',
          lr=0.0001,
          name='profis',
          print_progress=False):
    """
    Train the model
    :param model:
    :param train_loader:
    :param val_loader:
    :param annealer:
    :param epochs:
    :param device:
    :param lr:
    :param name:
    :param print_progress:
    :return:
    """

    charset = load_charset()
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
            if batch_idx % 50 == 0:
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

        if epoch % 50 == 0:
            # save the model
            torch.save(model.state_dict(), f'models/{name}/epoch_{epoch}.pt')

    return model

if __name__ == "__main__":

    # Parse the path to the config file
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', type=str, default='config_files/RNN_config.ini')
    args = argparser.parse_args()

    # Read the config file
    parser = configparser.ConfigParser()
    parser.read(args.config)

    # Initialize wandb
    wandb.init(project='profis2', name=parser['RUN']['run_name'])

    # Load the data and create the dataloaders
    fp_type = parser['MODEL']['fp_type']
    train_df = pd.read_parquet('data/RNN_dataset_ECFP_train_90.parquet' if fp_type == 'ECFP4' else
                               'data/RNN_dataset_KRFP_train_90.parquet')
    test_df = pd.read_parquet('data/RNN_dataset_ECFP_val_10.parquet' if fp_type == 'ECFP4' else
                              'data/RNN_dataset_KRFP_val_10.parquet')
    data_train = ProfisDataset(train_df, fp_len=int(parser['MODEL']['fp_len']))
    data_val = ProfisDataset(test_df, fp_len=int(parser['MODEL']['fp_len']))
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=int(parser['RUN']['batch_size']),
                                               shuffle=True, num_workers=int(parser['RUN']['num_workers']))
    val_loader = torch.utils.data.DataLoader(data_val,
                                             batch_size=int(parser['RUN']['batch_size']),
                                             shuffle=False, num_workers=int(parser['RUN']['num_workers']))
    torch.manual_seed(42)

    # Create a directory to save the models to
    if os.path.exists('models') is False:
        os.makedirs('models')
    model_name = parser['RUN']['run_name']
    if os.path.exists(f'models/{model_name}') is False:
        os.makedirs(f'models/{model_name}')
    else:
        raise ValueError(f'Model directory of the name {model_name} already exists')

    # Dump the config file to the model directory
    with open(f'models/{model_name}/config.ini', 'w') as f:
        parser.write(f)

    # Initialize the annealing agent
    annealer = Annealer(int(parser['RUN']['annealing_max_epoch']), parser['RUN']['annealing_shape'], baseline=0.0,
                        disable=parser['RUN'].getboolean('disable_annealing'))

    # Initialize the model
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if parser['RUN'].getboolean('use_cuda') else 'cpu'
    model = initialize_profis(args.config)
    model.to(device)

    model = train(model=model,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  epochs=int(parser['RUN']['epochs']),
                  annealer=annealer,
                  device=device,
                  lr=float(parser['RUN']['learn_rate']),
                  name=model_name,
                  print_progress=False)