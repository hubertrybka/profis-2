import math
import numpy as np
import torch
from configparser import ConfigParser
from profis.dataset import ProfisDataset, load_charset
from profis.net import Profis
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from torch.utils.data import DataLoader


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

def initialize_profis(config_path):
    config = ConfigParser()
    config.read(config_path)
    model = Profis(
        fp_size=int(config["MODEL"]["fp_len"]),
        fc1_size=int(config["MODEL"]["fc1_size"]),
        fc2_size=int(config["MODEL"]["fc2_size"]),
        hidden_size=int(config["MODEL"]["hidden_size"]),
        latent_size=int(config["MODEL"]["latent_size"]),
        gru_layers=int(config["MODEL"]["gru_layers"]),
        eps_coef=float(config["MODEL"]["eps_coef"]),
        dropout=float(config["MODEL"]["dropout"]),
        alphabet_size=len(load_charset()))
    return model

def is_valid(smiles):
    """
    Check if a SMILES string is valid
    :param smiles: SMILES string
    :return: True if valid, False otherwise
    """
    if Chem.MolFromSmiles(smiles, sanitize=True) is None:
        return False
    else:
        return True

def KRFP_score(mol, fp: torch.Tensor):
    """
    Calculates the KRFP fingerprint reconstruction score for a molecule
    Args:
        mol: rdkit mol object
        fp: torch tensor of size (fp_len)
    Returns:
        score: float (0-1)
    """
    score = 0
    key = pd.read_csv("data/KlekFP_keys.txt", header=None)
    fp_len = fp.shape[0]
    for i in range(fp_len):
        if fp[i] == 1:
            frag = Chem.MolFromSmarts(key.iloc[i].values[0])
            score += mol.HasSubstructMatch(frag)
    return score / torch.sum(fp).item()


def ECFP_score(mol, fp: torch.Tensor):
    """
    Calculates the ECFP fingerprint reconstruction score for a molecule
    Args:
        mol: rdkit mol object
        fp: torch tensor of size (fp_len)
    Returns:
        score: float (0-1)
    """
    score = 0
    fp_len = fp.shape[0]
    ECFP_reconstructed = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_len)
    for i in range(fp_len):
        if ECFP_reconstructed[i] and fp[i]:
            score += 1
    return score / torch.sum(fp).item()


def try_QED(mol):
    """
    Tries to calculate the QED score for a molecule
    Args:
        mol: rdkit mol object
    Returns:
        qed: float
    """
    try:
        qed = QED.qed(mol)
    except:
        qed = 0
    return qed

def smiles2sparse_KRFP(smiles):
    """
    Convert SMILES string to sparse Klekota&Roth fingerprint
    Args:
        smiles (str): SMILES string
    Returns:
        np.array: sparse fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    keys = "data/KlekFP_keys.txt"
    klek_keys = [line.strip() for line in open(keys)]
    klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
    fp_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            fp_list.append(1)
        else:
            fp_list.append(0)
    return np.array(fp_list)


def smiles2dense_KRFP(smiles):
    """
    Convert SMILES string to dense Klekota&Roth fingerprint
    Args:
        smiles (str): SMILES string
    Returns:
        np.array: dense fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    keys = "data/KlekFP_keys.txt"
    klek_keys = [line.strip() for line in open(keys)]
    klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
    fp_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            fp_list.append(i)
    return np.array(fp_list)


def sparse2dense(sparse, return_numpy=True):
    """
    Convert sparse fingerprint to dense fingerprint
    Args:
        sparse (np.array): sparse fingerprint
        return_numpy (bool): whether to return numpy array or list
    Returns:
        dense (np.array): dense fingerprint
    """
    dense = []
    for idx, value in enumerate(sparse):
        if value == 1:
            dense.append(idx)
    if return_numpy:
        return np.array(dense)
    else:
        return dense


def dense2sparse(dense, fp_len=4860):
    """
    Convert dense fingerprint to sparse fingerprint
    Args:
        dense (np.array): dense fingerprint
        fp_len (int): length of the fingerprint
    Returns:
        sparse (np.array): sparse fingerprint
    """
    sparse = np.zeros(fp_len, dtype=np.int8)
    for value in dense:
        sparse[value] = 1
    return np.array(sparse)


def encode(df, model, device, batch=1024):
    """
    Encode a dataframe containing FPs into latent space vectors
    :param df:
    :param model:
    :param device:
    :param batch:
    :return:
    """
    dataset = ProfisDataset(df, fp_len=model.fp_size)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloader:
            X, _ = batch
            X = X.to(device)
            mu, logvar = model.encode(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars


def smiles2sparse_ECFP(smiles, n_bits=2048):
    """
    Convert SMILES string to sparse ECFP fingerprint
    Args:
        smiles (str): SMILES string
        n_bits (int): number of bits in the fingerprint
    Returns:
        np.array: sparse fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    return np.array(fp)


def smiles2dense_ECFP(smiles, n_bits=2048):
    """
    Convert SMILES string to dense ECFP fingerprint
    Args:
        smiles (str): SMILES string
        n_bits (int): number of bits in the fingerprint
    Returns:
        np.array: dense fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    return sparse2dense(fp)