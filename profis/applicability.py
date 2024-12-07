import numpy as np
import pandas as pd
from torch import device
import torch
from profis.utils import encode
from profis.utils import initialize_profis
import configparser


class SCAvgMeasure:
    """
    Class to calculate the average cosine similarity of the n closest neighbors of a query molecule among the latent
    representations of the training set.
    Args:
        clf_path: path to the trained classifier model
    """

    def __init__(self, clf_path="models/D2_SMILES_KRFP_MLP/clf.pkl"):
        self.clf_path = clf_path
        self.config = self.read_config()
        self.train_path = self.config["RUN"]["data_path"]
        self.encoder_path = self.config["RUN"]["model_path"]
        self.device = device("cpu")
        self.encoder = initialize_profis(
            self.encoder_path.replace(
                self.encoder_path.split("/")[-1], "config.ini"
            )
        )
        self.encoder.load_state_dict(
            torch.load(self.encoder_path, map_location=self.device)
        )
        self.train_encoded = self.encode_fingerprints()

    def __call__(self, query, n=3):
        """
        Calculate the average cosine dissimilarity (distance) of the n closest neighbors of a query molecule among the
        latent representations of the training set compounds.
        Args:
            query: latent representation of the query molecule
            n: number of closest neighbors to consider
        Returns:
            scavg: average cosine dissimilarity of the n closest neighbors
        """
        distances = self.calc_cosa_to_train(query)

        # find n closest neighbors
        idx = np.argsort(distances)[::-1][:n]

        # calculate average similarity of n closest neighbors
        scavg = 1 - np.mean(distances[idx])
        return scavg

    def read_config(self):
        config = configparser.ConfigParser()
        config.read(self.clf_path.replace(self.clf_path.split("/")[-1], "config.ini"))
        return config

    def encode_fingerprints(self):
        data = pd.read_parquet(self.train_path)
        encoded = encode(data, self.encoder, device=self.device)[0]
        return encoded

    def calc_cosa_to_train(self, query):
        """
        Calculate the cosine similarity of the query molecule to all molecules in the training set.
        Args:
            query: latent representation of the query molecule
        Returns:
            distances: cosine similarity of the query molecule to all molecules in the training set
        """
        distances = np.dot(self.train_encoded, query) / (
            np.linalg.norm(self.train_encoded, axis=1) * np.linalg.norm(query)
        )
        return distances

