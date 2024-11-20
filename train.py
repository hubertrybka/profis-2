import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset
from rdkit import Chem
import wandb
import re
import argparse
from tqdm import tqdm
import os

class ProfisDataset(Dataset):
    """
    Dataset class for handling RNN training data.
    Parameters:
        df (pd.DataFrame): dataframe containing SMILES and fingerprints,
                           SMILES must be contained in ['smiles'] column as strings,
                           fingerprints must be contained in ['fps'] column as lists
                           of integers (dense vectors).
        vectorizer: vectorizer object instantiated from vectorizer.py
        fp_len (int): length of fingerprint
    """

    def __init__(self, df, fp_len=4860, smiles_enum=False):
        self.smiles = df["smiles"]
        self.fps = df["fps"]
        self.fps = self.prepare_X(self.fps)
        self.smiles = self.prepare_y(self.smiles)
        self.fp_len = fp_len
        self.smiles_enum = smiles_enum
        self.charset = load_charset()
        self.char2idx = {s: i for i, s in enumerate(self.charset)}

    def __getitem__(self, idx):
        """
        Get item from dataset.
        Args:
            idx (int): index of item to get
        Returns:
            X (torch.Tensor): reconstructed fingerprint
            y (torch.Tensor): vectorized SELFIES
        """
        raw_smile = self.smiles[idx]
        vectorized_seq = self.vectorize(raw_smile)
        if len(vectorized_seq) > 100:
            vectorized_seq = vectorized_seq[:100]
        raw_X = self.fps[idx]
        X = np.array(raw_X, dtype=int)
        X_reconstructed = self.reconstruct_fp(X)
        return (
            torch.from_numpy(X_reconstructed).float(),
            torch.from_numpy(vectorized_seq).float(),
        )

    def __len__(self):
        return len(self.fps)

    def reconstruct_fp(self, fp):
        fp_rec = np.zeros(self.fp_len)
        fp_rec[fp] = 1
        return fp_rec

    def prepare_X(self, fps):
        fps = fps.apply(lambda x: np.array(x, dtype=int))
        return fps.values

    @staticmethod
    def prepare_y(seq):
        return seq.values

    def vectorize(self, seq, pad_to_len=100):
        splited = self.split(seq) + ["[nop]"] * (pad_to_len - len(self.split(seq)))
        X = np.zeros((len(splited), len(self.charset)))

        for i in range(len(splited)):
            if splited[i] not in self.charset:
                raise ValueError(
                    f"Invalid token: {splited[i]}, allowed tokens: {self.charset}"
                )
            X[i, self.char2idx[splited[i]]] = 1
        return X

    def split(self, smile):
        pattern = (
            r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|"
            r"\*|\$|\%[0-9]{2}|[0-9]|[start]|[nop]|[end])")
        return re.findall(pattern, smile)

def one_hot_array(i, n):
    return map(int, [ix == i for ix in range(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def load_charset(path='data/smiles_alphabet.txt'):
    with open(path) as f:
        charset = f.readlines()
    charset = [char.strip() for char in charset]
    return charset

class Profis(nn.Module):
    def __init__(self,
                 fp_size=2048,
                 fc1_size=1024,
                 fc2_size=1024,
                 latent_size=32,
                 alphabet_size=len(load_charset())
        ):
        super(Profis, self).__init__()

        self.fc1 = nn.Linear(fp_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc31 = nn.Linear(fc2_size, latent_size)
        self.fc32 = nn.Linear(fc2_size, latent_size)
        self.fc4 = nn.Linear(latent_size, 256)
        self.gru = nn.GRU(256, 512, 3, batch_first=True)
        self.fc5 = nn.Linear(512, alphabet_size)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = self.selu(self.fc4(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 100, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = self.softmax(self.fc5(out_reshape))
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

def vae_loss(x_decoded_mean, y, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, y, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss, kl_loss

def is_valid(smiles):
    try:
        Chem.MolFromSmiles(smiles)
        return True
    except:
        return False

def train(model, train_loader, val_loader, epochs=100, device='cuda', lr=0.0001, print_progress=False):

    charset = load_charset()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print('Using device:', device)

    for epoch in range(1, epochs + 1):

        print(f'Epoch', epoch)

        model.train()
        train_loss = 0
        mean_kld_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader)) if print_progress else enumerate(train_loader):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output, mean, logvar = model(X)
            loss, kld_loss = vae_loss(output, y, mean, logvar)
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

    return model

argparser = argparse.ArgumentParser()
argparser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train the model')
argparser.add_argument('--batch_size', type=int, default=128)
argparser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate for the optimizer')
argparser.add_argument('--fp_type', type=str, default='ECFP',
                       help='Type of fingerprint to use (ECFP or KRFP)',
                       choices=['ECFP', 'KRFP'])
argparser.add_argument('--name', type=str, default='profis')

args = argparser.parse_args()

wandb.init(project='profis2', name=args.name)

train_df = pd.read_parquet('data/RNN_dataset_ECFP_train_90.parquet' if args.fp_type == 'ECFP' else
                           'data/RNN_dataset_KRFP_train_90.parquet')
test_df = pd.read_parquet('data/RNN_dataset_ECFP_val_10.parquet' if args.fp_type == 'ECFP' else
                          'data/RNN_dataset_KRFP_val_10.parquet')
data_train = ProfisDataset(train_df, fp_len=2048 if args.fp_type == 'ECFP' else 4860)
data_val = ProfisDataset(test_df, fp_len=2048 if args.fp_type == 'ECFP' else 4860)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

torch.manual_seed(42)

epochs = args.epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Profis(fp_size=2048 if args.fp_type == 'ECFP' else 4860).to(device)
model = train(model, train_loader, val_loader, epochs, device, lr=args.lr, print_progress=True)

if os.path.exists('models') is False:
    os.makedirs('models')

model.save(f'models/{args.name}.pt')