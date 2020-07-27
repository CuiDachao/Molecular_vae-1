# -------------------------------------------------- IMPORTS --------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import gc
import sys
import re
from rdkit import Chem
from standardiser import standardise
from sklearn.utils import shuffle

from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- PROCESS DATASETS --------------------------------------------------

prism_zinc = []

prism_metadata = pd.read_csv("/hps/research1/icortes/acunha/data/Drug_Sensitivity_PRISM/primary-screen-replicate-collapsed-treatment-info.csv", sep=',', header=0, index_col=0,  usecols=['column_name', 'smiles'], nrows = 50)
prism_metadata.dropna(subset=['smiles'], inplace=True)
smiles = list(prism_metadata['smiles'])

del prism_metadata
gc.collect()

with open('/hps/research1/icortes/acunha/data/ZINC/250k_rndm_zinc_drugs_clean.smi', "r") as f:
    zinc_smiles = f.readlines()
    
smiles.extend(zinc_smiles)
for smile in smiles:
    smile = smile.strip("\n")
    if ',' in smile: #means that exists more than one smile representation of the compound
        if '"' in smile:
            smile = smile.strip('"')
        smile = smile.split(", ")
    else:
        smile = [smile]
    prism_zinc.extend(smile)

for item in [smiles, zinc_smiles]:
    del item
gc.collect()
        
prism_zinc = shuffle(prism_zinc)

print("Before standardiser: {}".format(len(prism_zinc)))

standard_smiles = []
for i in range(len(prism_zinc)):
    smile = prism_zinc[i]
    try:
        m = Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True, canonical=True)
        mol = standardise.run(m)
        standard_smiles.append(mol)
    except standardise.StandardiseException:
        pass

print("After standardiser: {}".format(len(standard_smiles)))

del prism_zinc
gc.collect()

with open('/hps/research1/icortes/acunha/data/ZINC_PRISM_SMILES/zinc_prism_smiles_processed.smi', "w") as f:
    f.write('\n'.join(standard_smiles))