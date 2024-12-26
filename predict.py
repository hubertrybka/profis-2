import argparse
import configparser
import os
import time

import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
import torch

from profis.pred import predict, filter_dataframe
from profis.utils import initialize_profis


def main(config_path):
    """
    Generates structure predictions for the latent embeddings of molecular fingerprints.
    Args:
        config_path: Path to the config file.
    """

    start_time = time.time()

    # get config
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_path)
    file_path = config["RUN"]["data_path"]
    model_path = config["RUN"]["model_path"]
    use_cuda = config["RUN"].getboolean("use_cuda")
    clf_data_path = config["RUN"]["clf_data_path"]
    verbosity = int(config["RUN"]["verbosity"])
    batch_size = int(config["RUN"]["batch_size"])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    if clf_data_path and not os.path.exists(clf_data_path):
        raise FileNotFoundError(f"Classifier train dataset {clf_data_path} not found")

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Using {device} device") if verbosity > 0 else None

    # get file name
    dirname = os.path.dirname(file_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    model_config_path = model_path.replace(model_path.split("/")[-1], "config.ini")
    model_config = configparser.ConfigParser(allow_no_value=True)
    if not os.path.exists(model_config_path):
        raise ValueError(f"Model config file {model_config_path} not found")
    model_config.read(model_config_path)
    out_encoding = model_config["RUN"]["out_encoding"]

    # load model

    model = initialize_profis(config_path=model_config_path).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}") if verbosity >= 1 else None

    # load data
    if file_path.endswith(".csv"):
        query_df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        query_df = pd.read_parquet(file_path)
    else:
        raise ValueError("Data file format not supported (must be .csv or .parquet)")

    for col in ["smiles", "label", "score", "activity", "norm", "distance_to_model"]:
        if col in query_df.columns:
            query_df = query_df.drop(columns=[col])
    input_vector = query_df.to_numpy()
    print(f"Loaded data from {file_path}") if verbosity >= 1 else None

    # get predictions
    print(f"Getting predictions for file {file_path}...") if verbosity >= 1 else None
    df = predict(
        model,
        input_vector,
        device=device,
        encoding_format=out_encoding,
        batch_size=batch_size,
        dropout=config["RUN"].getboolean("dropout"),
        show_progress=config["RUN"].getboolean("show_progress"),
    )

    # filter dataframe
    if len(df) > 0:
        df = filter_dataframe(df, config, verbose=(True if verbosity > 0 else False))
    else:
        print("No valid predictions") if verbosity > 0 else None

    # save stats
    stats = pd.DataFrame()
    stats["mean_qed"] = df["qed"].mean()

    # save data as csv
    os.mkdir(f"{dirname}/preds_{timestamp}")
    with open(f"{dirname}/preds_{timestamp}/config.ini", "w") as configfile:
        config.write(configfile)
    df.to_csv(f"{dirname}/preds_{timestamp}/predictions.csv", index=False)

    (
        print(f"Saved data to {dirname}/preds_{timestamp} directory")
        if verbosity > 0
        else None
    )

    # dump config
    with open(f"{dirname}/preds_{timestamp}/config.ini", "w") as configfile:
        config.write(configfile)

    # save images
    os.mkdir(f"{dirname}/preds_{timestamp}/imgs")
    for n, (idx, smiles) in enumerate(zip(df["idx"], df["smiles"])):
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(
            mol, f"{dirname}/preds_{timestamp}/imgs/{idx}_{n}.png", size=(300, 300)
        )

    time_elapsed = time.time() - start_time
    print(f"{file_path} processed in {(time_elapsed / 60):.2f} minutes")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config_files/pred_config.ini",
        help="Path to config file",
    )

    args = parser.parse_args()
    config_path = args.config

    main(config_path=config_path)
