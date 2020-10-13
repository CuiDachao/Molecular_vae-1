# -------------------------------------------------- IMPORTS --------------------------------------------------
import pandas as pd
import numpy as np
import pickle
from standardiser import standardise
import gc
from rdkit.Chem import AllChem
from rdkit import Chem

from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- RUN --------------------------------------------------

def process_data(type_dataset, dataset):
    new_ids = []
    fingerprints = []
    smiles = []
    
    n_smiles = 0
    for smile in dataset:
        try:
            mol = standardise.run(smile)
            if len(mol) <= 120:
                fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024)
                new_ids.append('{}_{}'.format(type_dataset, n_smiles + 1))
                fingerprints.append('[{}]'.format(','.join([str(x) for x in fp])))
                smiles.append(mol)
                n_smiles += 1
        except:
            pass
    
    with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/READ_ME.txt', 'a') as f:
        f.write('From {}: {} smiles\n'.format(type_dataset, n_smiles))
    
    return new_ids, fingerprints, smiles

# -------------------------------------------------- RUN --------------------------------------------------

path_data = '/hps/research1/icortes/acunha/data'
ohf = OneHotFeaturizer()

new_ids = []
fingerprints = []
smiles = []
datasets = []

with open('{}/ChEMBL_data/Chembl_250K_molecules.txt'.format(path_data), 'r') as f:
    chembl_compounds = f.readlines()
    chembl_compounds = [x.strip('\n') for x in chembl_compounds]

ids, fp, smi = process_data('chembl_compounds', chembl_compounds)
# res_oh = {}
# for i in range(len(smi)):
#     whole_dataset_oh = ohf.featurize(smi, 120)
#     for j in range(len(whole_dataset_oh)):
#         if str(whole_dataset_oh[j]) != 'nan':
#             res_oh[ids[j]] = whole_dataset_oh[j]
# pickle.dump(res_oh, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250_onehot::chembl_compounds.pkl', 'wb'))
new_ids.extend(ids)
fingerprints.extend(fp)
smiles.extend(smi)
datasets.extend(['chembl_compounds' for _ in range(len(smi))])

# to_delete = [whole_dataset_oh, chembl_compounds, new_ids, fp, smi]
# for item in to_delete:
#     del item
# gc.collect()

chembl_approved_drugs = pd.read_csv('{}/ChEMBL_approved_drugs/Chembl_approved_drugs.csv'.format(path_data), header = 0, index_col = 0)
chembl_approved_drugs = list(chembl_approved_drugs['smiles'])
ids, fp, smi = process_data('chembl_approved_drugs', chembl_approved_drugs)
# res_oh = {}
# for i in range(len(smi)):
#     whole_dataset_oh = ohf.featurize(smi, 120)
#     for j in range(len(whole_dataset_oh)):
#         if str(whole_dataset_oh[j]) != 'nan':
#             res_oh[ids[j]] = whole_dataset_oh[j]
# pickle.dump(res_oh, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250_onehot::chembl_approved_drugs.pkl', 'wb'))
new_ids.extend(ids)
fingerprints.extend(fp)
smiles.extend(smi)
datasets.extend(['chembl_approved_drugs' for _ in range(len(smi))])
# 
# to_delete = [whole_dataset_oh, chembl_approved_drugs, new_ids, fp, smi]
# for item in to_delete:
#     del item
# gc.collect()

with open('{}/PRISM_every_compound/Prism_all.txt'.format(path_data), 'r') as f:
    prism = f.readlines()
    prism = [x.strip('\n') for x in prism]
ids, fp, smi = process_data('prism', prism)
# res_oh = {}
# for i in range(len(smi)):
#     whole_dataset_oh = ohf.featurize(smi, 120)
#     for j in range(len(whole_dataset_oh)):
#         if str(whole_dataset_oh[j]) != 'nan':
#             res_oh[ids[j]] = whole_dataset_oh[j]
# pickle.dump(res_oh, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250_onehot::prism.pkl', 'wb'))
new_ids.extend(ids)
fingerprints.extend(fp)
smiles.extend(smi)
datasets.extend(['prism' for _ in range(len(smi))])

# to_delete = [whole_dataset_oh, prism, new_ids, fp, smi]
# for item in to_delete:
#     del item
# gc.collect()
    
with open('{}/ZINC/250k_rndm_zinc_drugs_clean.smi'.format(path_data), 'r') as f:
    zinc = f.readlines()
    zinc = [x.strip('\n') for x in zinc]
ids, fp, smi = process_data('zinc', zinc)
# res_oh = {}
# for i in range(len(smi)):
#     whole_dataset_oh = ohf.featurize(smi, 120)
#     for j in range(len(whole_dataset_oh)):
#         if str(whole_dataset_oh[j]) != 'nan':
#             res_oh[ids[j]] = whole_dataset_oh[j]
# pickle.dump(res_oh, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250_onehot::zinc.pkl', 'wb'))
new_ids.extend(ids)
fingerprints.extend(fp)
smiles.extend(smi)
datasets.extend(['zinc' for _ in range(len(smi))])

# to_delete = [whole_dataset_oh, zinc]
# for item in to_delete:
#     del item
# gc.collect()

res = pd.DataFrame(new_ids, columns = ['index'])
res['Morgan_Fingerprint'] = fingerprints
res['Smile'] = smiles
res['Dataset'] = datasets
res.to_csv('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.txt',
                         header=True, index=False)
pickle.dump(res, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.pkl', 'wb'))
res['Smile'].to_csv('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250_Smile.txt',
                         header=False, index=False)

print('done!')