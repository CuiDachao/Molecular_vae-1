# -------------------------------------------------- IMPORTS --------------------------------------------------
import pickle

from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- FUNCTIONS --------------------------------------------------

def check_valid_smiles(data, maximum_length):
    found_invalid = False
    valid_smiles = []
    for i in range(len(data)):
        m = data[i]
        if len(m) <= maximum_length and m not in valid_smiles:
            valid_smiles.append(m)
        else:
            found_invalid = True
            with open("invalid_smiles.txt", 'a') as f:
                f.write(data[i])
                f.write('\n')

    if found_invalid:
        print('WARNING!! \nSome molecules have invalid lengths and will not be considered. Please check the file invalid_smiles.txt for more information. \n')

    return valid_smiles

# -------------------------------------------------- RUN --------------------------------------------------

ohf = OneHotFeaturizer()
maximum_length = 120
with open('/hps/research1/icortes/acunha/data/ZINC/250k_rndm_zinc_drugs_clean.smi', 'r') as f:
    whole_dataset = []
    for smi in f:
        whole_dataset.append(smi.strip("\n"))

whole_dataset = check_valid_smiles(whole_dataset[:100], maximum_length)
valid_smiles = len(whole_dataset)

whole_dataset_oh = ohf.featurize(whole_dataset, maximum_length)

whole_dataset_smiles_final = []
whole_dataset_onehot_final = []
for i in range(len(whole_dataset)):
    if str(whole_dataset_oh[i]) != 'nan':
        whole_dataset_smiles_final.append(whole_dataset[i])
        whole_dataset_onehot_final.append(whole_dataset_oh[i])


pickle.dump(whole_dataset_smiles_final, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/zinc/zinc_final_smiles.pkl', 'wb'))
pickle.dump(whole_dataset_onehot_final, open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/zinc/zinc_final_onehot.pkl', 'wb'))
with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/zinc/READ_ME.txt', 'w') as f:
    f.write('Number of valid compounds (before converting into onehot matrices): {} \n'.format(valid_smiles))
    f.write('Number of one hot encoded matrices: {}'.format(len(whole_dataset_onehot_final)))