# -------------------------------------------------- IMPORTS --------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import pandas as pd
import pickle
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch.utils.data
import sys
import gc
from rdkit import Chem

from VAE_NN import Molecular_VAE
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open("/hps/research1/icortes/acunha/python_scripts/vae_smiles_prism_zinc/check_valid/{}".format(filename), 'a') as f:
        f.write('\n'.join(list_comments))
    
# --------------------------------------------------

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
        print(
            'WARNING!! \nSome molecules have invalid lengths and will not be considered. Please check the file invalid_smiles.txt for more information. \n')

    return valid_smiles

# -------------------------------------------------- GENEXP --------------------------------------------------

class Molecular():

    def __init__(self):
        self.alpha = None
        self.maximum_length = None
        self.size_batch = None
        self.dropout = None
        self.seed = None

        self.ohf = None
        self.device = None
        
        self.filename_report = None
        
    # --------------------------------------------------

    def __set_parameters(self, list_parameters):
        self.alpha = float(list_parameters[0])
        self.maximum_length = int(list_parameters[1])
        self.size_batch = int(list_parameters[3])
        self.dropout = float(list_parameters[7])
        self.seed = int(list_parameters[10])
        self.ohf = OneHotFeaturizer()
        
        #add information to report
        lines = ["** REPORT - MOLECULAR **\n",
                "* Parameters",
                "Alpha (1.0 is without alpha): {}".format(self.alpha),
                "Maximum length os smiles: {}".format(self.maximum_length),
                "Size batch: {} ; Dropout: {}".format(self.size_batch, self.dropout),
                "Seed: {} \n".format(self.seed)]
        create_report(self.filename_report, lines)

    # --------------------------------------------------

    def __load_initial_parameters(self, path):
        list_parameters = pickle.load(open('{}/pickle/list_initial_parameters_smiles.pkl'.format(path), 'rb'))
        self.__set_parameters(list_parameters)
        self.device = list_parameters[-1]

    # --------------------------------------------------

    def __initialize_model(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        model = Molecular_VAE(number_channels_in=self.maximum_length,
                              length_signal_in=len(self.ohf.get_charset()), dropout_prob = self.dropout)
        model.to(self.device)
        
        lines = ["\n*About the network",
                "Runs on: {} \n".format(self.device)]
        create_report(self.filename_report, lines)
        
        return model

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output, z_mu, z_var):
        reconstruction_loss = F.binary_cross_entropy(x_output, x_input, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        return reconstruction_loss + (self.alpha * kl_loss), reconstruction_loss, kl_loss

    # --------------------------------------------------

    def __load_model(self, model, path):
        model_parameters = pickle.load(open('{}/pickle/molecular_model.pkl'.format(path), 'rb'))
        model.load_state_dict(model_parameters)
        return model
    
    # --------------------------------------------------

    def start_molecular(self, path):
        self.__load_initial_parameters(path)
        model = self.__initialize_model(n_rows=self.maximum_length, n_columns=len(ohf.get_charset()))
        model = self.__load_model(model, path)
        return model
    
    # --------------------------------------------------
    
    def run_new_latent_space(self, model_trained, dataset, times):
        dataset_torch = torch.tensor(dataset).type('torch.FloatTensor')
        data_loader = torch.utils.data.DataLoader(dataset_torch, batch_size=1, shuffle=False)
        
        del dataset_torch
        gc.collect()
        
        total_loss = 0.0
        reconstruction_loss = 0.0
        kl_loss = 0.0
        predictions_complete, bottleneck_complete = [], []
        model_trained.eval()
        with torch.no_grad():
            for data_batch in data_loader:
                number_valid = 0
                data_batch = data_batch.to(self.device)
                z_mu, z_var = model_trained.encoder(data_batch)
                correct_seed = copy.copy(self.seed)
                
                j = 0
                while j < times:
                    std = torch.exp(z_var/2)
                    eps = torch.randn_like(std) * 1e-2
                    x_sample = eps.mul(std).add_(z_mu)
                    data_predictions = model_trained.decoder(x_sample)  # output predicted by the model
                    smile_o = ohf.back_to_smile(list(data_predictions))
                    
                    try:
                        m = Chem.MolFromSmiles(smile_o[0])
                        if m is not None:
                            number_valid += 1
                    except:
                        pass
                    
                    self.seed = np.random.random_integers(0, 10000)
                    j += 1
                
                self.seed = correct_seed
                
                lines = ["Smile: {}".format(ohf.back_to_smile(list(data_batch))),
                         "Number of valid smiles after {} iterations: {}".format(times, number_valid),
                         "\n"]
                create_report(self.filename_report, lines)
                

        free_memory = [data_loader, data_predictions]
        for item in free_memory:
            del item
        gc.collect()
    
    # --------------------------------------------------
    
    def get_maximum_length(self):
        return self.maximum_length
        
    # --------------------------------------------------

    def set_filename_report(self, filename):
        self.filename_report = filename

# -------------------------------------------------- RUN --------------------------------------------------

molecules = Molecular()
molecules.set_filename_report("molecular_output_50times.txt")
path = "/hps/research1/icortes/acunha/python_scripts/vae_smiles_prism_zinc/results/0.00005/120_0.00001_64_2000_0.7_0.15_0_0.9_100_500"
mol_model = molecules.start_molecular(path)
maximum_length_smiles = int(molecules.get_maximum_length())

prism_metadata = pd.read_csv("/hps/research1/icortes/acunha/data/Drug_Sensitivity_PRISM/primary-screen-replicate-collapsed-treatment-info.csv", sep=',', header=0, index_col=0,  usecols=['column_name', 'smiles'], nrows = 50)
prism_metadata.dropna(subset=['smiles'], inplace=True)
smiles = list(prism_metadata['smiles'])

del prism_metadata
gc.collect()

prism = []
for smile in smiles:
    smile = smile.strip("\n")
    if ',' in smile: #means that exists more than one smile representation of the compound
        if '"' in smile:
            smile = smile.strip('"')
        smile = smile.split(", ")
    else:
        smile = [smile]
    prism.extend(smile)

del smiles
gc.collect()

standard_smiles = []
for i in range(len(prism)):
    smile = prism[i]
    try:
        m = Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=True, canonical=True)
        mol = standardise.run(m)
        standard_smiles.append(mol)
    except standardise.StandardiseException:
        pass
    
del prism
gc.collect()

list_smiles = ohf.featurize(standard_smiles, maximum_length_smiles)
print(len(list_smiles))
list_smiles = [x for x in list_smiles if str(x) != 'nan']
print(list_smiles)
molecules.run_new_latent_space(mol_model, list_new_smiles, 50)