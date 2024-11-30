import math
import re
import numpy as np
import torch
from torch.utils.data import Dataset

class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return

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

class Smiles2SmilesDataset(Dataset):
    """
    Dataset class for handling RNN training data.
    Parameters:
        df (pd.DataFrame): dataframe containing SMILES and fingerprints,
                           SMILES must be contained in ['smiles'] column as strings,
                           fingerprints must be contained in ['fps'] column as lists
                           of integers (dense vectors).
    """

    def __init__(self, df):
        self.smiles = df["smiles"].values
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
        return torch.from_numpy(vectorized_seq).float()

    def __len__(self):
        return len(self.smiles)


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