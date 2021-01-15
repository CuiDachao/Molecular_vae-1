# -------------------------------------------------- IMPORTS --------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import torch.utils.data
import h5py
from rdkit import Chem
import random
import seaborn as sns

from VAE_NN import Molecular_VAE
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- ANOTHER FUNCTIONS --------------------------------------------------

def set_seed(value):
    global seed
    np.random.seed(value)
    torch.manual_seed(value)

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

initial_seed = 42
set_seed(initial_seed)

# -------------------------------------------------- MOLECULAR --------------------------------------------------

path_model = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model'
# path_model = '/Users/acunha/Desktop/Molecular_VAE'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, maximum_length_m, _, _, _, _, _, dropout_m, _, _, _, _, _, _ = pickle.load(open('{}/list_initial_parameters_smiles.pkl'.format(path_model), 'rb'))
dropout_m = float(dropout_m)
maximum_length_m = int(maximum_length_m)
ohf = OneHotFeaturizer()

molecular_model = Molecular_VAE(number_channels_in=maximum_length_m, length_signal_in=len(ohf.get_charset()), dropout_prob=dropout_m)
molecular_model.load_state_dict(torch.load('{}/molecular_model.pt'.format(path_model), map_location=device))
molecular_model.to(device)


# path_dataset = path_model
path_dataset = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/'
dataset = h5py.File('{}/prism_chembl250_chembldrugs_zinc250.hdf5'.format(path_dataset), 'r')

# path_results = path_dataset
path_results = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/final_results'
test_set_indexes = open('{}/Test_indexes.txt'.format(path_results), 'r')
test_set_indexes = test_set_indexes.readlines()
test_set_indexes = [x.strip('\n') for x in test_set_indexes]
print(len(test_set_indexes))

seeds = [initial_seed]
new_dataset = {}
results = {}
list_number_epochs = [1, 50, 100, 500]
number_epochs = list_number_epochs[-1]
while len(seeds) < number_epochs:
    x = random.randint(0, 100000)
    if x not in seeds:
        seeds.append(x)
for i in range(len(test_set_indexes)):
    index = test_set_indexes[i]
    position = np.where(np.char.decode(dataset['index']) == index)[0][0]
    smile = np.char.decode(dataset['smiles'][position]).tolist()
    j = 1
    found_valid = False
    found_same = False
    valid = -1
    same = -1
    input_tensor = torch.Tensor([np.array(dataset['one_hot_matrices'][position])]).type('torch.FloatTensor').to(device)
    z_mu, z_var = molecular_model.encoder(input_tensor)
    while j <= number_epochs or found_same:
        print(j)
        set_seed(seeds[j-1])
        std = torch.exp(z_var/2)
        eps = torch.randn_like(std) * 1e-2
        x_sample = eps.mul(std).add_(z_mu)
        output = molecular_model.decoder(x_sample)
        smile_output = ohf.back_to_smile(output)[0]
        print(smile_output)
        m = Chem.MolFromSmiles(smile_output)
        if m is not None and not found_valid:
            valid = j
            found_valid = True
        if smile == smile_output:
            same = j
            found_same = True
        j += 1

    new_dataset[index] = {'Number_for_valid':valid, 'Number_for_identical':same}

new_dataset = pd.DataFrame.from_dict(new_dataset, orient='index')
new_dataset.to_csv('{}/Results_epoch{}.csv'.format(path_results, number_epochs), header=True, index=True)

number_per_epochs = {}
number_per_epochs['>{}'.format(number_epochs)] = {'Valid':new_dataset.loc[new_dataset['Number_for_valid'] == -1].shape[0],
                                                  'Identical':new_dataset.loc[new_dataset['Number_for_identical'] == -1].shape[0]}
for epoch in range(1, number_epochs+1):
    number_per_epochs[str(epoch)] = {
        'Valid': new_dataset.loc[new_dataset['Number_for_valid'] == epoch].shape[0],
        'Identical': new_dataset.loc[new_dataset['Number_for_identical'] == epoch].shape[0]}

number_per_epochs = pd.DataFrame.from_dict(number_per_epochs, orient='index').rename_axis('Epochs')
subset = number_per_epochs.reset_index()
subset = pd.melt(subset, id_vars = "Epochs")
subset.sort_values(by=['Epochs'], inplace=True)
print(subset)

my_palette = ["#17202a", "#641e16", "#012f11"]
sns.set_palette(my_palette)
sns.set_style(style='white')

sns.catplot(x='Epochs', y='value', hue='variable', data=subset, kind = 'bar')._legend.remove()
plt.ylabel('Number of cases')
plt.legend()
plt.savefig('{}/Number_molecules_per_epoch.png'.format(path_results), dpi=200, bbox_inches='tight')
plt.show()

results = {}
for number in list_number_epochs:
    indexes = [x for x in range(1, number+1)]
    subset = number_per_epochs.loc[number_per_epochs.index.isin(indexes)]
    results['With {} epochs'.format(number)] = {'Valid':subset['Valid'].sum(),
                                                'Identical':subset['Identical'].sum(),
                                                'Not found':new_dataset.shape[0] - subset['Valid'].sum()}
results = pd.DataFrame.from_dict(results, orient='index')
print(results)