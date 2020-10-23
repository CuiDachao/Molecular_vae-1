# -------------------------------------------------- IMPORTS --------------------------------------------------
import pandas as pd
import h5py
import numpy as np
import pickle
from standardiser import standardise
import gc
from rdkit.Chem import AllChem
from rdkit import Chem

from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- RUN --------------------------------------------------

def process_data(type_dataset, dataset, i):
    new_ids = []
    fingerprints = []
    smiles = []
    mol_oh = []
    
    n_smiles = 0
    for smile in dataset:
        try:
            mol = standardise.run(smile)
            if len(mol) <= 120:
                fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024)
                m_oh = ohf.featurize([mol], 120)
                if str(m_oh) != 'nan':
                    new_ids.append('{}_{}'.format(type_dataset, n_smiles + 1 + i))
                    fingerprints.append('[{}]'.format(','.join([str(x) for x in fp])))
                    smiles.append(mol)
                    mol_oh.append(m_oh[0])
                    n_smiles += 1
        except:
            print('{} in {}'.format(smile, type_dataset))
            pass
    
    print(len(smiles))
    
    return np.string_(new_ids), np.string_(fingerprints), np.string_(smiles), np.array(mol_oh), n_smiles + i

# -------------------------------------------------- RUN --------------------------------------------------

path_data = '/hps/research1/icortes/acunha/data/'
ohf = OneHotFeaturizer()

with open('{}/ChEMBL_data/Chembl_250K_molecules.txt'.format(path_data), 'r') as f:
    whole_dataset = f.readlines()
    whole_dataset = [x.strip('\n') for x in whole_dataset]
whole_dataset = list(set(whole_dataset))
for i in range(0, len(whole_dataset), 500):
    ids, fp, smi, mol_oh, n_smiles = process_data('chembl_compounds', whole_dataset[i:i+500], i)
    datasets = np.string_(['chembl_compounds'] * ids.shape[0])
    d = {'index': ids, 'morgan_fingerprints': fp, 'smiles': smi, 'one_hot_matrices': mol_oh, 'datasets': datasets}
    if i == 0:
        with h5py.File('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.hdf5', 'w') as f:
            f.create_dataset('index', data = ids, compression="gzip", chunks=True, maxshape=(None,))
            f.create_dataset('morgan_fingerprints', data = fp, compression="gzip", chunks=True, maxshape=(None,))
            f.create_dataset('smiles', data = smi, compression="gzip", chunks=True, maxshape=(None,))
            f.create_dataset('one_hot_matrices', data = mol_oh, compression="gzip", chunks=True, maxshape=(None,None,None))
            f.create_dataset('datasets', data = datasets, compression="gzip", chunks=True, maxshape=(None,))
    else:
        with h5py.File('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.hdf5', 'a') as f:
            for k in d.keys():
                array = d[k]
                print(f[k].shape[0] + array.shape[0])
                f[k].resize((f[k].shape[0] + array.shape[0]), axis = 0)
                f[k][-array.shape[0]:] = array
    print('chembl_compounds, ', i)

with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/READ_ME.txt', 'a') as f:
    f.write('From chembl_compounds: {} smiles\n'.format(n_smiles))

whole_dataset = pd.read_csv('{}/ChEMBL_approved_drugs/Chembl_approved_drugs.csv'.format(path_data), header = 0, index_col = 0)
whole_dataset = list(set(whole_dataset['smiles']))
for i in range(0, len(whole_dataset), 500):
    ids, fp, smi, mol_oh, n_smiles = process_data('chembl_approved_drugs', whole_dataset[i:i+500], i)
    datasets = np.string_(['chembl_approved_drugs'] * ids.shape[0])
    d = {'index': ids, 'morgan_fingerprints': fp, 'smiles': smi, 'one_hot_matrices': mol_oh, 'datasets': datasets}
    with h5py.File('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.hdf5', 'a') as f:
        for k in d.keys():
            array = d[k]
            f[k].resize((f[k].shape[0] + array.shape[0]), axis = 0)
            f[k][-array.shape[0]:] = array
    print('chembl_approved_drugs, ', i)

with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/READ_ME.txt', 'a') as f:
    f.write('From chembl_approved_drugs: {} smiles\n'.format(n_smiles))

with open('{}/PRISM_every_compound/Prism_all.txt'.format(path_data), 'r') as f:
    whole_dataset = f.readlines()
    whole_dataset = [x.strip('\n') for x in whole_dataset]
whole_dataset = list(set(whole_dataset))
for i in range(0, len(whole_dataset), 500):
    ids, fp, smi, mol_oh, n_smiles = process_data('prism', whole_dataset[i:i+500], i)
    print(ids.shape[0])
    datasets = np.string_(['prism'] * ids.shape[0])
    d = {'index': ids, 'morgan_fingerprints': fp, 'smiles': smi, 'one_hot_matrices': mol_oh, 'datasets': datasets}
    with h5py.File('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.hdf5', 'a') as f:
        for k in d.keys():
            array = d[k]
            f[k].resize((f[k].shape[0] + array.shape[0]), axis = 0)
            f[k][-array.shape[0]:] = array
    print('prism, ', i)

with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/READ_ME.txt', 'a') as f:
    f.write('From prism: {} smiles\n'.format(n_smiles))
    
with open('{}/ZINC/250k_rndm_zinc_drugs_clean.smi'.format(path_data), 'r') as f:
    whole_dataset = f.readlines()
    whole_dataset = [x.strip('\n') for x in whole_dataset]
whole_dataset = list(set(whole_dataset))
for i in range(0, len(whole_dataset), 500):
    ids, fp, smi, mol_oh, n_smiles = process_data('zinc', whole_dataset[i:i+500], i)
    datasets = np.string_(['zinc'] * ids.shape[0])
    d = {'index': ids, 'morgan_fingerprints': fp, 'smiles': smi, 'one_hot_matrices': mol_oh, 'datasets': datasets}
    with h5py.File('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.hdf5', 'a') as f:
        for k in d.keys():
            array = d[k]
            f[k].resize((f[k].shape[0] + array.shape[0]), axis = 0)
            f[k][-array.shape[0]:] = array
    print('zinc, ', i)

with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/READ_ME.txt', 'a') as f:
    f.write('From zinc: {} smiles\n'.format(n_smiles))

with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.txt', 'w') as f:
    f.write('index,Morgan_Fingerprint,Smile\n')
    with h5py.File('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.hdf5', 'r') as file:
        d1 = file['index']
        d2 = file['morgan_fingerprints']
        d3 = file['smiles']
        d4 = file['datasets']
        for i in range(len(d1)):
            f.write('{},{},{},{}\n'.format(np.char.decode(d1[i]).tolist(), np.char.decode(d2[i]).tolist(),
                                           np.char.decode(d3[i]).tolist(), np.char.decode(d4[i]).tolist()))

print('done!')