import torch
import numpy as np
from profis.utils import decode_seq_from_indexes, load_charset, initialize_profis, smiles2sparse_KRFP
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from scipy.spatial.distance import jaccard

#suppress rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def is_valid(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return False
    return True

def forward_w_fp_noise(fp, model, n_bits=10, n_tries=1000, device="cuda"):

    noised = []
    for _ in range(n_tries):
        idxs = np.random.choice(np.arange(len(fp)), n_bits, replace=False)
        # switch bits on/off
        noisy_fp = fp.copy()
        noisy_fp[idxs] = 1 - noisy_fp[idxs]
        noised.append(noisy_fp)
    noised = torch.tensor(noised, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(noised)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    outputs = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader):
            X = data[0]  # get only X (fingerprint)
            # add some random noise to the fingerprint (switch bits on/off)
            noisy_X = X.clone()
            # draw n_bits random indexes
            idxs = np.random.choice(np.arange(4860), n_bits, replace=False)
            # switch bits on/off
            noisy_X[:, idxs] = 1 - noisy_X[:, idxs]
            X = noisy_X.to(device)
            output, _, _ = model(X)
            outputs.append(output.detach().cpu())

    outputs = torch.cat(outputs, dim=0).numpy()
    charset = load_charset()
    out_smiles = [
        decode_seq_from_indexes(out.argmax(axis=1), charset)
        for out in outputs
    ]
    out_smiles = [smile.replace("[nop]", "") for smile in out_smiles]
    validity = sum([is_valid(smile) for smile in out_smiles]) / len(out_smiles)
    return out_smiles, validity

def get_tanimoto_similarities(ref_smiles, out_smiles):
    fp = smiles2sparse_KRFP(ref_smiles)
    tanimotos = []
    for smile in out_smiles:
        try:
            fp2 = smiles2sparse_KRFP(smile)
            tanimoto = jaccard(fp2, fp)
            tanimotos.append(tanimoto)
        except:
            tanimotos.append(0)
    return tanimotos

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = initialize_profis(f'models/KRFP_SMILES/config.ini').to(device)
model.load_state_dict(torch.load(f'models/KRFP_SMILES/epoch_600.pt'))

drugs = {
    'sulpiride': 'CCN1CCCC1CNC(=O)C1=C(OC)C=CC(=C1)S(N)(=O)=O',
    'clozapine': 'CN1CCN(CC1)C2=C3C=CC=CC3=NC4=C(N2)C=C(C=C4)Cl',
    'haloperidol': 'C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F'
}

ks = [1, 2, 5, 10, 15, 20]

for key, smiles in drugs.items():
    print(key)
    print(smiles)
    fp = smiles2sparse_KRFP(smiles)
    df = pd.DataFrame()
    for k in ks:
        out_smiles, validity = forward_w_fp_noise(fp, model, n_bits=k, n_tries=10000, device="cuda")
        print(f'k={k}')
        print(f'Validity: {validity}')
        tanimotos = get_tanimoto_similarities(smiles, out_smiles)
        df[f'{k}_bits'] = tanimotos
    df.to_csv(f'{key}_tanimotos.csv')