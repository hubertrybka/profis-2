import os
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# gnina path
GNINA = "/home/bmc/gnina/gnina"


def run_gnina(
    smiles,
    name="test",
    gnina_path=GNINA,
    num_cpu=-1,
    verbose=True,
    no_gpu=False,
    cnn_scoring=False,
    num_modes=5,
    dirname="gnina_out",
):
    """
    Run gnina on a dictionary of conformers in .sdf format.
    :param smiles:
    :param name:
    :param gnina_path:
    :param num_cpu:
    :param verbose:
    :param no_gpu:
    :param cnn_scoring:
    :param num_modes:
    :param dirname:
    :return:
    """
    start_time = time.time()

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Adds hydrogens to make optimization more accurate
        AllChem.EmbedMolecule(mol)  # Adds 3D positions
        AllChem.MMFFOptimizeMolecule(
            mol
        )  # Improves the 3D positions using a force-field method
    except Exception:
        raise

    Chem.MolToMolFile(mol, f"tmp/{name}.mol")

    os.system(f"obabel tmp/{name}.mol -O tmp/{name}.pdbqt -p")

    print(f"(gnina) Docking {name}...") if verbose else None

    cmd = (
        f"{gnina_path} -r receptor.pdb -l tmp/{name}.pdbqt "
        + "--autobox_ligand placeholder_ligand.pdb --autobox_add 10 "
        + f"--num_modes {num_modes} --cpu {num_cpu} -o gnina_out_{filename}/{name}.sdf --exhaustiveness 16"
    )

    if not cnn_scoring:
        cmd += " --cnn_scoring none"

    if no_gpu:
        cmd += " --no_gpu"

    os.system(cmd + f" > gnina_out_{dirname}/{name}.log")

    time_elapsed = round(time.time() - start_time, 3)

    with open(f"gnina_out_{dirname}/{name}.log") as file:
        output = file.read()
        output = parse_gnina_output(output)
        top = get_top_score(output, score="affinity")

    # remove temp files
    if os.path.exists(f"tmp/{name}.mol"):
        os.system(f"rm tmp/{name}.mol")
    if os.path.exists(f"tmp/{name}.pdbqt"):
        os.system(f"rm tmp/{name}.pdbqt")

    return top, round(time_elapsed, 3)


def return_table_idcs(file):
    """
    Return indices of first line of each table in gnina output file.
    Input:
        file: str, gnina output file
    """
    lines = file.split("\n")
    table_idcs = []

    for i, line in enumerate(lines):
        if line.startswith("----"):
            table_idcs.append(i + 1)

    return table_idcs


def extract_table(file, idx):
    """
    Extract table from gnina output file.
    Input:
        file: str, gnina output file
        idx: int, index of first line of table
    Returns:
        table: pd.DataFrame, table of gnina output
    """

    lines = file.split("\n")
    table = pd.DataFrame()
    while True:
        try:
            values = lines[idx].split()
            values_dict = {
                "mode": values[0],
                "affinity": values[1],
                "CNN_score": values[2],
                "CNN_affinity": values[3],
            }
            int(values_dict["mode"])
        except IndexError:
            break
        except ValueError:
            break
        table = pd.concat([table, pd.DataFrame(values_dict, index=[0])])
        idx += 1
    table = table.astype(
        {"mode": int, "affinity": float, "CNN_score": float, "CNN_affinity": float}
    )
    assert len(table) > 0, "Table is empty!"
    return table


def parse_gnina_output(file):
    """
    Extract all tables from gnina output file and return as one concatenated table.
    Input:
        file: str, gnina output file
    Returns:
        big_table: pd.DataFrame, concatenated table of gnina output
    """
    big_table = pd.DataFrame()
    table_idxs = return_table_idcs(file)
    tables = []
    for idx in table_idxs:
        tables.append(extract_table(file, idx))

    if len(tables) > 1:
        for n, table in enumerate(tables):
            table_tmp = table.copy()
            table_tmp["table"] = [n] * len(table)
            big_table = pd.concat([big_table, table_tmp])
        big_table = big_table.reset_index(drop=True)
    else:
        big_table = tables[0]

    return big_table


def get_top_score(table, score="CNN_affinity"):
    """
    Return top (affinity, CNN_score or CNN_affinity) value from gnina output file (default: CNN_affinity).
    Input:
        table: pd.DataFrame, table of gnina output
    Returns:
        top_score: float, top CNN_score
    """
    if score == "affinity":
        top_score = table[score].min()
    elif score == "CNN_score" or score == "CNN_affinity":
        top_score = table[score].max()
    else:
        raise ValueError("score must be affinity, CNN_score or CNN_affinity")
    return top_score


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to .smi file")
    parser.add_argument(
        "--num_modes", type=int, help="number of modes to generate", default=5
    )
    parser.add_argument("--num_cpu", type=int, help="number of CPUs to use", default=-1)
    parser.add_argument(
        "--cnn_scoring", action="store_true", help="use CNN scoring", default=False
    )
    parser.add_argument(
        "--no_gpu", action="store_true", help="run gnina on CPU only", default=True
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print progress", default=False
    )
    parser.add_argument(
        "--get_top_score",
        action="store_true",
        help="get top scores as .csv",
        default=False,
    )
    args = parser.parse_args()

    filename = args.path.split("/")[-1].split(".")[0]

    # create tmp folder
    os.mkdir(f"gnina_out_{filename}")
    print("Directory ", f"gnina_out_{filename}", " created.")
    if not os.path.exists(f"tmp"):
        os.mkdir(f"tmp")

    if args.num_cpu == -1:
        num_cpu = os.cpu_count()
        print(f"Using {num_cpu} CPUs")
    else:
        num_cpu = args.num_cpu

    start_time = time.time()

    results_df = pd.DataFrame(columns=["smiles", "score"])
    df = pd.read_csv(f"{args.path}", sep="\t", names=["smiles", "name"])
    smiles_list = df["smiles"].apply(lambda x: x.strip("\n")).tolist()

    times = []

    for i, zipped in enumerate(zip(df["name"].tolist(), smiles_list)):
        name, smiles = zipped
        try:
            top_score, single_docking_time = run_gnina(
                smiles,
                name=name,
                num_cpu=num_cpu,
                no_gpu=args.no_gpu,
                verbose=args.verbose,
                cnn_scoring=args.cnn_scoring,
                num_modes=args.num_modes,
                dirname=filename,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        times.append(single_docking_time)
        mean_time = round(np.array(times).mean(), 2)

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)

        print("+------------------------------+")
        print(f"Mean docking time: {mean_time}")
        print(f"Executed at: {current_time}")
        print("+------------------------------+")
        results = pd.DataFrame(
            {"smiles": smiles, "score": top_score, "time": single_docking_time},
            index=[i],
        )
        if results_df.shape[0] == 0:
            results_df = results
        else:
            results_df = pd.concat([results_df, results])
            results_df.to_csv(f"gnina_out_{filename}/scores.csv", index=False)
    time_elapsed = time.time() - start_time
    print(f"Finished in {time_elapsed / 60:.2f} minutes")
