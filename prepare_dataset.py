import pandas as pd
import argparse
import rdkit.Chem as Chem
import os
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
from profis.utils import sparse2dense, smiles2sparse_KRFP, smiles2sparse_ECFP
from profis.utils.vectorizer import SMILESVectorizer


def prepare(data_path, gen_ecfp=False, gen_krfp=False, to_dense=True):
    """
    Prepare the dataset by standardizing the SMILES strings, checking for compatibility with
    the model's token alphabet and (optionally) generating fingerprints.
    Args:
        data_path (str): Path to the dataset.
        gen_ecfp (bool): Generate ECFP fingerprints.
        gen_krfp (bool): Generate KRFP fingerprints.
        to_dense (bool): Convert sparse fingerprints to dense fingerprints.
    """

    # handle input
    if gen_ecfp and gen_krfp:
        raise ValueError("Please choose only one fingerprint type to generate")

    # Load the dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} not found")
    df = pd.read_csv(data_path)

    # Check the column names
    df.columns = df.columns.str.lower()
    if "fps" in df.columns and (gen_ecfp or gen_krfp):
        raise ValueError("Fingerprint column already exists in the dataset")

    for name in ["smiles", "activity", "fps"]:
        if name not in df.columns:
            raise ValueError(f"Column {name} not found in the dataset")

    print(f"Loaded data from {data_path}")

    # Standardize the SMILES strings and check for validity
    df["smiles"] = df["smiles"].apply(try_standardize_smiles)
    df = df.dropna(subset=["smiles"])

    # Check for compatibility with the model's token alphabet
    vectorizer = SMILESVectorizer()
    df["is_compatible"] = df["smiles"].apply(
        check_if_alphabet_compatibile, vectorizer=vectorizer
    )
    df = df[df["is_compatible"]]

    # Generate fingerprints
    if gen_ecfp:
        df["fps"] = df["smiles"].apply(smiles2sparse_ECFP)
    elif gen_krfp:
        df["fps"] = df["smiles"].apply(smiles2sparse_KRFP)

    # Convert sparse fingerprints to dense fingerprints
    if to_dense or gen_ecfp or gen_krfp:
        df["fps"] = df["fps"].apply(sparse2dense)

    # Save the processed dataset
    out_path = data_path.replace(".parquet", "_processed.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Processed data saved to {out_path}")

    return


def try_standardize_smiles(smiles):
    """
    Standardize the SMILES string. Returns None if the SMILES string is invalid or cannot be standardized.
    Args:
        smiles (str): SMILES string.
    Returns:
        str: Standardized SMILES string.
    """
    u = rdMolStandardize.Uncharger()
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = rdMolStandardize.Cleanup(mol)
        uncharged_mol = u.uncharge(mol)
    except:
        return None
    return Chem.MolToSmiles(uncharged_mol)


def check_if_alphabet_compatibile(smiles, vectorizer):
    """
    Check if the SMILES string is compatible with the model's token alphabet.
    Args:
        smiles (str): SMILES string.
        vectorizer (SMILESVectorizer): SMILES vectorizer.
    Returns:
        bool: True if the SMILES string is compatible, False otherwise.
    """
    try:
        vectorizer.vectorize(smiles)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--gen_ecfp", action="store_true", help="Generate ECFP fingerprints"
    )
    parser.add_argument(
        "--gen_krfp", action="store_true", help="Generate KRFP fingerprints"
    )
    parser.add_argument(
        "--to_dense",
        action="store_true",
        help="Convert sparse fingerprints to dense fingerprints",
    )
    args = parser.parse_args()
    prepare(args.data_path, args.skip_fingerprint)
