import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from profis.utils import smiles2sparse_ECFP
import argparse


class TanimotoSearch:
    """
    Class for calculating the minimal Tanimoto distance between a query molecule and a dataset of molecules.
    Parameters:
        data_path (str): path to the dataset
    """

    def __init__(self, data_path, verbose=False):
        self.data_path = data_path
        self.smiles = pd.read_parquet(data_path)["smiles"]
        self.XB = np.array([smiles2sparse_ECFP(x, 512) for x in self.smiles]).reshape(
            -1, 512
        )
        self.verbose = verbose

    def __call__(self, smiles, return_similar=False):
        XA = smiles2sparse_ECFP(smiles, 512).reshape(1, -1)
        dists = cdist(XA, self.XB, "jaccard")
        min_dist = np.min(dists)
        top1_similar_smiles = self.smiles[np.argmin(dists)]
        (
            print(
                f"Minimal Tanimoto distance: {round(min_dist, 3)}"
                if self.verbose
                else None
            )
            if self.verbose
            else None
        )
        if return_similar:
            return min_dist, top1_similar_smiles
        else:
            return min_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot molecular properties")
    parser.add_argument(
        "-d", "--data_path", type=str, help="Path to the classifier training set"
    )
    parser.add_argument(
        "-s", "--smiles", type=str, help="SMILES string of the molecule to search for"
    )
    args = parser.parse_args()
    search = TanimotoSearch(args.data_path, verbose=False)
