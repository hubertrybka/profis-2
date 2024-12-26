import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from profis.utils import initialize_profis
import deepsmiles as ds
import selfies as sf
import json
import numpy as np
from rdkit.Chem import Draw

from rdkit import Chem
from rdkit import RDLogger
rdlogger = RDLogger.logger()
rdlogger.setLevel(RDLogger.CRITICAL)

def draw_from_latent(model_path):

    model_dir = '/'.join(model_path.split('/')[:-1])
    encoding_format = model_dir.split('/')[-1].split('_')[1]
    fp_type = model_dir.split('/')[-1].split('_')[0]
    distrib = model_dir + '/aggregated_posterior.json'
    with open(distrib) as f:
        mus_sigmas = json.load(f)

    n_samples = 10000
    mus = np.array(mus_sigmas['mean'])
    sigmas = np.array(mus_sigmas['std'])
    samples = np.random.normal(mus, sigmas, (n_samples, mus.shape[0]))
    normal_samples = np.random.normal(0, 1, (n_samples, mus.shape[0]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = initialize_profis(model_dir + '/config.ini').to(device)
    model.load_state_dict(torch.load(model_dir + '/epoch_600.pt'))

    prior_samples = decode_random_samples(model, normal_samples, encoding_format, device)
    posterior_samples = decode_random_samples(model, samples, encoding_format, device)

    prior_mols = [Chem.MolFromSmiles(s) for s in prior_samples]
    posterior_mols = [Chem.MolFromSmiles(s) for s in posterior_samples]
    prior_mols = [m for m in prior_mols if m is not None]
    posterior_mols = [m for m in posterior_mols if m is not None]

    print_mols(prior_mols, fp_type, encoding_format, 'prior')
    print_mols(posterior_mols, fp_type, encoding_format, 'posterior')

    prior_valid = len(prior_mols) / n_samples
    posterior_valid = len(posterior_mols) / n_samples

    return prior_valid, posterior_valid

def decode_random_samples(model, samples, encoding_format, device):
    dataset = TensorDataset(torch.tensor(samples).float())
    # dataloader
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    # put the model in evaluation mode
    model.eval()
    # list to store the generated samples
    outputs = []
    # generate the samples
    with torch.no_grad():
        for batch in tqdm(dataloader):
            z = batch[0].to(device)
            sample = model.decode(z)
            outputs.append(sample.cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)

    # decode the samples to SMILES
    from profis.dataset import load_charset
    from profis.utils import decode_seq_from_indexes

    charset = load_charset(f'data/{encoding_format.lower()}_alphabet.txt')

    smiles_strings = []
    converter = ds.Converter(rings=True, branches=True)
    for output in outputs:
        argmaxed = np.argmax(output, axis=1)
        seq = (decode_seq_from_indexes(argmaxed, charset)).replace('[nop]', '')
        if encoding_format == 'SELFIES':
            smiles = sf.decoder(seq)
        elif encoding_format == 'DeepSMILES':
            try:
                smiles = converter.decode(seq)
            except:
                smiles = None
        elif encoding_format == 'SMILES':
            smiles = seq
        else:
            raise ValueError('Unknown encoding format')
        smiles_strings.append(smiles)
    smiles_strings = [s for s in smiles_strings if s is not None]
    return smiles_strings

def print_mols(mols, fp_type, encoding_format, distribution):
    #draw 30 random mols
    d2d = Draw.MolDraw2DCairo(250,200)
    dopts = d2d.drawOptions()
    dopts.drawMolsSameScale = False

    mols_to_draw = np.random.choice(mols, 40)
    img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=5, subImgSize=(400, 400), returnPNG=False, drawOptions=dopts)
    img.save(f'imgs/{fp_type}_{encoding_format}_sampled_mols_{distribution}.png')

if __name__ == '__main__':

    models = ['models/ECFP_SMILES/epoch_600.pt',
              'models/ECFP_DeepSMILES/epoch_600.pt',
              'models/KRFP_SMILES/epoch_600.pt',
              'models/KRFP_DeepSMILES/epoch_600.pt',]
    for model in models:
        prior_valid, posterior_valid = draw_from_latent(model)

        print(f'Model: {model}')
        print(f'Prior validity: {prior_valid*100:.2f} %')
        print(f'Posterior validity: {posterior_valid*100:.2f} %')