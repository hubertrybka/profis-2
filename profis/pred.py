from operator import index

import numpy as np
import pandas as pd
import rdkit.Chem.Crippen as Crippen
import torch
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors
from torch.utils.data import DataLoader
from profis.tanimoto import TanimotoSearch
from profis.dataset import load_charset
from profis.utils import decode_seq_from_indexes

# Suppress RDKit warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def predict(
    model,
    latent_vectors: np.array,
    device: torch.device = torch.device("cpu"),
    encoding_format: str = "smiles",
    batch_size: int = 64,
    dropout=False,
):
    """
    Generate molecules from latent vectors
    Args:
        model (torch.nn.Module): ProfisGRU model.
        latent_vectors (np.array): numpy array of latent vectors. Shape = (n_samples, latent_size).
        device: device to use for prediction. Can be 'cpu' or 'cuda'.
        encoding_format: encoding format of the output. Can be 'smiles', 'selfies' or 'deepsmiles'.
        batch_size: batch size for prediction.

    Returns:
        pd.DataFrame: Dataframe containing smiles and scores.
    """

    device = torch.device(device)

    loader = DataLoader(latent_vectors, batch_size=batch_size, shuffle=False)

    if not dropout:
        model.eval()
    with torch.no_grad():
        df = pd.DataFrame(columns=["idx", "smiles"])
        preds_list = []
        for X in loader:
            latent_tensor = torch.Tensor(X).type(torch.FloatTensor).to(device)
            preds = model.decode(latent_tensor)
            preds = preds.detach().cpu().numpy()
            preds_list.append(preds)
        preds_concat = np.concatenate(preds_list)

        if encoding_format == "smiles":
            charset = load_charset()
            for i in range(len(preds_concat)):
                argmaxed = preds_concat[i].argmax(axis=1)
                smiles = decode_seq_from_indexes(argmaxed, charset).replace("[nop]", "")
                row = pd.DataFrame({"idx": i, "smiles": smiles}, index=[len(df)])
                df = pd.concat([df, row])
        else:
            raise ValueError(
                "Invalid encoding format. Can be 'smiles', 'deepsmiles' or 'selfies'."
            )

        df["idx"] = range(len(df))

    return df


def get_largest_ring(mol):
    """
    Returns the size of the largest ring in a molecule.
    Args:
        mol (rdkit.Chem.Mol): Molecule object.
    """
    ri = mol.GetRingInfo()
    rings = []
    for b in mol.GetBonds():
        ring_len = [len(ring) for ring in ri.BondRings() if b.GetIdx() in ring]
        rings += ring_len
    return max(rings) if rings else 0


def try_sanitize(mol):
    """
    Tries to sanitize a molecule object. If sanitization fails, returns the original molecule.
    Args:
        mol (rdkit.Chem.Mol): Molecule object.
    """
    try:
        output = mol
        Chem.SanitizeMol(output)
        return output
    except:
        return mol


def try_smiles2mol(smiles):
    """
    Tries to convert a SMILES string to a molecule object. If conversion fails, returns None.
    """
    try:
        output = Chem.MolFromSmiles(smiles)
        return output
    except:
        return None


def apply_filter(df, column, func, min_val=None, max_val=None, verbose=False):
    """
    Applies a filter to a DataFrame based on a calculated column.

    Args:
        df (pd.DataFrame): DataFrame to filter.
        column (str): Name of the new column to store calculated values.
        func (callable): Function to calculate the column values.
        min_val (float or int, optional): Minimum threshold for filtering.
        max_val (float or int, optional): Maximum threshold for filtering.
        verbose (bool): Whether to print debug information.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df[column] = df["mols"].apply(func)
    if min_val is not None:
        df = df[df[column] >= min_val]
    if max_val is not None:
        df = df[df[column] <= max_val]
    if verbose:
        print(f"Number of molecules after filtering by {column}: {len(df)}")
    return df


def filter_dataframe(df, config, verbose=False):
    """
    Filters a DataFrame of molecules based on the given configuration.

    Args:
        df (pd.DataFrame): DataFrame containing molecules.
        config (dict): Dictionary containing filtering parameters.
        verbose (bool): Whether to print debug information.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df_copy = df.copy()
    df_copy["mols"] = df_copy["smiles"].apply(Chem.MolFromSmiles)
    df_copy.dropna(axis=0, inplace=True, subset=["mols"])

    # Define filter configurations
    filters = [
        (
            "largest_ring",
            get_largest_ring,
            config["RING_SIZE"].getint("min"),
            config["RING_SIZE"].getint("max"),
        ),
        (
            "num_rings",
            Chem.rdMolDescriptors.CalcNumRings,
            config["NUM_RINGS"].getint("min"),
            config["NUM_RINGS"].geint("max"),
        ),
        ("qed", QED.default, config["QED"].getfloat("min"), config["QED"].getfloat("max")),
        (
            "mol_wt",
            Chem.rdMolDescriptors.CalcExactMolWt,
            config["MOL_WEIGHT"].getfloat("min"),
            config["MOL_WEIGHT"].getfloat("max"),
        ),
        (
            "num_HBA",
            rdMolDescriptors.CalcNumHBA,
            config["NUM_HBA"].getint("min"),
            config["NUM_HBA"].getint("max"),
        ),
        (
            "num_HBD",
            rdMolDescriptors.CalcNumHBD,
            config["NUM_HBD"].getint("min"),
            config["NUM_HBD"].getint("max"),
        ),
        ("logP", Crippen.MolLogP, config["LOGP"].getfloat("min"), config["LOGP"].getfloat("max")),
        (
            "num_rotatable_bonds",
            rdMolDescriptors.CalcNumRotatableBonds,
            config["NUM_ROT_BONDS"].getint("min"),
            config["NUM_ROT_BONDS"].getint("max"),
        ),
        (
            "tpsa",
            rdMolDescriptors.CalcTPSA,
            config["TPSA"].getfloat("min"),
            config["TPSA"].getfloat("max"),
        ),
        (
            "bridgehead_atoms",
            rdMolDescriptors.CalcNumBridgeheadAtoms,
            config["NUM_BRIDGEHEAD_ATOMS"].getint("min"),
            config["NUM_BRIDGEHEAD_ATOMS"].getint("max"),
        ),
        (
            "spiro_atoms",
            rdMolDescriptors.CalcNumSpiroAtoms,
            config["NUM_SPIRO_ATOMS"].getint("min"),
            config["NUM_SPIRO_ATOMS"].getint("max"),
        ),
    ]

    # Apply each filter
    for column, func, min_val, max_val in filters:
        df_copy = apply_filter(df_copy, column, func, min_val, max_val, verbose=verbose)

    # Handle novelty score separately
    if config["RUN"].get("clf_data_path"):
        ts = TanimotoSearch(config["RUN"]["clf_data_path"])
        tanimoto_search_results = df_copy["smiles"].apply(
            lambda x: ts(x, return_similar=True)
        )
        df_copy["novelty_score"] = tanimoto_search_results.apply(lambda x: x[0])
        df_copy["closest_in_train"] = tanimoto_search_results.apply(lambda x: x[1])

        df_copy = apply_filter(
            df_copy,
            "novelty_score",
            lambda x: x,
            config["NOVELTY_SCORE"].getfloat("min"),
            config["NOVELTY_SCORE"].getfloat("max"),
            verbose=verbose,
        )
    else:
        if verbose:
            print("Skipping novelty filtering: clf_data_path not provided.")

    # Drop redundant columns
    df_copy.drop(columns=["mols"], inplace=True)

    return df_copy
