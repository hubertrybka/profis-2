import argparse
import configparser
import os
import pickle
import time
import json

import numpy as np

import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from profis.dataset import Smiles2SmilesDataset
from profis.net import MolecularVAE
from torch.utils.data import DataLoader
from profis.clf import nested_CV


def main(config_path, verbose=True):
    """
    Trains a QSAR classifier on the latent embeddings of the known ligands.
    Args:
        config_path (str): path to the config file
        verbose (bool): whether to print the progress
    """

    # read config file
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)

    model_type = determine_model_type(config)
    data_path = str(config["RUN"]["data_path"])
    model_path = str(config["RUN"]["model_path"])
    out_path = str(config["RUN"]["output_dir"])
    name = str(config["RUN"]["name"])
    optimize = config.getboolean("RUN", "optimize_hyperparameters")

    start_time = time.time()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file {config_path} not found")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    cuda_available = torch.cuda.is_available() and config.getboolean("RUN", "use_cuda")
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Using device: {device}") if verbose else None

    if out_path is None or not os.path.exists(f"{out_path}"):
        out_path = "models"

    if not os.path.exists(f"{out_path}/{name}"):
        os.mkdir(f"{out_path}/{name}")

    # read dataset

    data = pd.read_parquet(data_path, columns=["smiles", "activity", "fps"])
    data.reset_index(drop=True, inplace=True)
    print(f"Loaded data from {data_path}") if verbose else None
    activity = data["activity"]

    big_model = MolecularVAE().to(device)

    print(f"Loading weights from {model_path}") if verbose else None
    big_model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = Smiles2SmilesDataset(data)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

    mus = []
    big_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch.to(device)
            mu, _ = big_model.encode(X)
            mus.append(mu.cpu().numpy())
        mus = np.concatenate(mus, axis=0)

    data = pd.DataFrame(mus)
    data["activity"] = activity
    data.reset_index(drop=True, inplace=True)
    X = data.drop("activity", axis=1).values
    y = data["activity"].values

    # initialize the classifier

    if model_type == "SVC":
        params = {
            "C": float(config["SVC"]["c_param"]),
            "kernel": str(config["SVC"]["kernel"]),
            "gamma": str(config["SVC"]["gamma"]),
            "shrinking": True,
            "probability": True,
            "max_iter": -1,
        }
        with open("data/SVM_param_grid.json", "r") as f:
            param_grid = json.load(f)
        clf = SVC(**params)

    elif model_type == "RF":
        max_depth = (
            None
            if config["RF"]["max_depth"] == "None"
            else int(config["RF"]["max_depth"])
        )
        max_features = (
            None
            if config["RF"]["max_features"] == "None"
            else str(config["RF"]["max_features"])
        )
        max_leaf_nodes = (
            None
            if config["RF"]["max_leaf_nodes"] == "None"
            else int(config["RF"]["max_leaf_nodes"])
        )
        params = {
            "n_estimators": int(config["RF"]["n_estimators"]),
            "max_depth": max_depth,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "random_state": 42,
            "n_jobs": -1,
        }
        with open("data/RF_param_grid.json", "r") as f:
            param_grid = json.load(f)
        clf = RandomForestClassifier(**params)

    elif model_type == "XGB":
        params = {
            "learning_rate": float(config["XGB"]["learning_rate"]),
            "n_estimators": int(config["XGB"]["n_estimators"]),
            "max_depth": int(config["XGB"]["max_depth"]),
            "min_child_weight": int(config["XGB"]["min_child_weight"]),
            "gamma": float(config["XGB"]["gamma"]),
            "subsample": float(config["XGB"]["subsample"]),
            "nthread": int(config["XGB"]["nthread"]),
            "seed": 42,
            "random_state": 42,
        }
        with open("data/XGB_param_grid.json", "r") as f:
            param_grid = json.load(f)
        clf = XGBClassifier(**params)

    elif model_type == "MLP":
        fc1 = config["MLP"]["fc1"]
        fc2 = config["MLP"]["fc2"]
        network_size = [int(n) for n in [fc1, fc2] if n != ""]
        params = {
            "hidden_layer_sizes": network_size,
            "activation": str(config["MLP"]["activation"]),
            "solver": str(config["MLP"]["optimizer"]),
            "alpha": float(config["MLP"]["alpha"]),
            "learning_rate_init": float(config["MLP"]["learning_rate"]),
            "random_state": 42,
            "max_iter": 1000,
        }
        with open("data/MLP_param_grid.json", "r") as f:
            param_grid = json.load(f)
        clf = MLPClassifier(**params)

    else:
        raise ValueError(
            f"Model type {model_type} not recognized. The config file may be corrupted."
        )

    # hyperparameter optimization and cross-validation

    best_model, accuracy_scores, roc_auc_scores = nested_CV(
        clf, X, y, param_grid, optimize=optimize, verbose=verbose
    )
    clf = best_model

    # refit the model with the best hyperparameters
    clf.fit(X, y)
    best_params = clf.get_params()

    if verbose:
        (
            print(f"Best hyperparameters: {best_params}")
            if optimize
            else print(f"Hyperparameters: {best_params}")
        )
        print(
            f"Accuracy: {round(accuracy_scores.mean(), 4)} +/- {round(accuracy_scores.std(), 4)}"
        )
        print(
            f"ROC_AUC: {round(roc_auc_scores.mean(), 4)} +/- {round(roc_auc_scores.std(), 4)}"
        )
    if optimize:
        with open(f"./{out_path}/{name}/best_params.txt", "w") as file:
            file.write(str(best_params))

    # save model

    with open(f"./{out_path}/{name}/clf.pkl", "wb") as file:
        pickle.dump(clf, file)

    metrics = {
        "accuracy": round(accuracy_scores.mean(), 4),
        "accuracy_std": round(accuracy_scores.std(), 4),
        "roc_auc": round(roc_auc_scores.mean(), 4),
        "roc_auc_std": round(roc_auc_scores.std(), 4),
    }
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f"{out_path}/{name}/metrics.csv", index=False)

    # dump config
    with open(f"{out_path}/{name}/config.ini", "w") as configfile:
        config.write(configfile)

    time_elapsed = round((time.time() - start_time), 2)
    if time_elapsed < 60:
        print(f"Finished in {time_elapsed} seconds")
    else:
        print(f"Finished in {round(time_elapsed / 60, 2)} minutes")
    return


def determine_model_type(config: configparser.ConfigParser):
    """
    Determine the model type from the config file.
    Args:
        config (ConfigParser): ConfigParser object.
    Returns:
        str: Model type.
    """
    detected_sections = [
        section
        for section in config.sections()
        if section in ["SVC", "RF", "XGB", "MLP"]
    ]
    if len(detected_sections) == 1:
        return detected_sections[0]
    else:
        raise ValueError("Model type not recognized. The config file may be corrupted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config_files/SVC_config.ini",
        help="Path to config file",
    )
    args = parser.parse_args()
    config_path = args.config
    main(config_path)
