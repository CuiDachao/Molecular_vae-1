# -------------------------------------------------- IMPORTS --------------------------
------------------------

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

from full_network import VAE_molecular
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- DEFINE SEEDS ---------------------
-----------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- ANOTHER FUNCTIONS ----------------
----------------------------------

def create_report(filename, list_comments):
    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/{}'.format
(filename), 'a') as f:
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
            with open('invalid_smiles.txt', 'a') as f:
                f.write(data[i])
                f.write('\n')

    if found_invalid:
        print('WARNING!! \nSome molecules have invalid lengths and will not be considered. Please check the file invalid_smiles.txt for more information. \n')

    return valid_smiles

# -------------------------------------------------- MOLECULAR --------------------------------------------------
class Molecular():
    def __init__(self):
        self.alpha = None
        self.size_input = None
        self.maximum_length = None
        self.size_batch = None
        self.dropout = None
        self.seed = None

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

    # --------------------------------------------------

    def __load_initial_parameters(self):
        list_parameters = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/list_initial_parameters_molecular.pkl', 'rb'))
        self.__set_parameters(list_parameters)
        self.device = list_parameters[-1]

        lines = ['** REPORT - MOLECULAR **\n',
                '* Parameters',
                'Alpha (1.0 is without alpha): {}'.format(self.alpha),
                'Maximum length os smiles: {}'.format(self.maximum_length),
                'Size batch: {} ; Dropout: {} ; Seed: {} '.format(self.size_batch, self.dropout, self.seed),
                '\n*About the network',
                'Runs on: {}'.format(self.device),
                '\n']

        create_report(self.filename_report, lines)

        global seed
        if seed != self.seed:
            seed = self.seed
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    # --------------------------------------------------

    def __initialize_model(self, n_rows, n_columns):
        model = VAE_molecular(number_channels_in=int(n_rows),
                              length_signal_in=int(n_columns), dropout_prob=self.dropout)
        model.to(self.device)

        return model

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output, z_mu, z_var):
        reconstruction_loss = F.binary_cross_entropy(x_output, x_input, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        return reconstruction_loss + (self.alpha * kl_loss), reconstruction_loss, kl_loss

    # --------------------------------------------------

    def __load_model(self, model):
        model_parameters = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/trained_models/molecular/molecular_model.pkl', 'rb'))
        model.load_state_dict(model_parameters)
        return model
    
    # --------------------------------------------------

    def start_molecular(self):
        self.__load_initial_parameters()
        model = self.__initialize_model(n_rows=self.maximum_length, n_columns=len(self.ohf.get_charset()))
        model = self.__load_model(model)
        return model
    
    # --------------------------------------------------

    def run_dataset(self, model_trained, dataset):
        dataset_torch = torch.tensor(dataset).type('torch.FloatTensor')
        data_loader = torch.utils.data.DataLoader(dataset_torch, batch_size=self.size_batch, shuffle=False)
        
        del dataset_torch
        gc.collect()
        
        total_loss = 0.0
        reconstruction_loss = 0.0
        kl_loss = 0.0
        predictions_complete, bottleneck_complete = [], []
        model_trained.eval()
        with torch.no_grad():
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                data_predictions = model_trained(data_batch)  # output predicted by the model
                current_loss = self.__loss_function(data_batch, data_predictions[0], data_predictions[2], data_predictions[3])
                total_loss += current_loss[0].item()
                reconstruction_loss += current_loss[1].item()
                kl_loss += current_loss[2].item()
                predictions_complete.extend(list(data_predictions[0].cpu().numpy()))
                bottleneck_complete.extend(list(data_predictions[1].cpu().numpy()))

        total_loss = total_loss / len(data_loader)
        reconstruction_loss = reconstruction_loss / len(data_loader)
        kl_loss = kl_loss / len(data_loader)

        free_memory = [data_loader, data_predictions]
        for item in free_memory:
            del item
        gc.collect()

        valid = 0
        invalid_id = []
        with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_once/valid_smiles.txt', 'a') as f:
            smiles_i = self.ohf.back_to_smile(list(dataset))
            smiles_o = self.ohf.back_to_smile(list(predictions_complete))
            for i in range(len(smiles_i)):
                m = Chem.MolFromSmiles(smiles_o[i])
                if m is not None:
                    valid += 1
                else:
                    invalid_id.append(i)
                f.write('\n'.join(['Input: {}'.format(smiles_i[i]), 'Output: {}'.format(smiles_o[i]), '\n']))
                f.write('\n')
        
        lines = ['* RUN JUST ONCE *',
                'Number of smiles: {} '.format(len(dataset)),
                'Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(total_loss, reconstruction_loss, kl_loss),
                'Valid molecules: {:.2f}%'.format(float((valid / len(smiles_o)) * 100)),
                '\n']
        create_report(self.filename_report, lines)
        
        return predictions_complete, bottleneck_complete, invalid_id
    
    # --------------------------------------------------
    def run_only_valids(self, model_trained, dataset, times, list_indexes):
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
            number_valid = 0
            valid_id = []
            for i, data_batch in enumerate(data_loader):
                data_batch = data_batch.to(self.device)
                valid = False
                z_mu, z_var = model_trained.encoder(data_batch)
                correct_seed = copy.copy(self.seed)
                
                j = 0
                while j < times and not valid:
                    std = torch.exp(z_var/2)
                    eps = torch.randn_like(std) * 1e-2
                    x_sample = eps.mul(std).add_(z_mu)
                    data_predictions = model_trained.decoder(x_sample)  # output predicted by the model
                    smile_o = self.ohf.back_to_smile(list(data_predictions))
                    
                    try:
                        m = Chem.MolFromSmiles(smile_o[0])
                        if m is not None:
                            valid = True
                            with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_{}/Results_valid.txt'.format(times), 'a') as f:
                                lines = ['Smile: {}'.format(self.ohf.back_to_smile(list(data_batch))),
                                         'Index: {}'.format(list_indexes[i]),
                                         'Found valid smile after {} iterations'.format(j+1),
                                         '\n']
                                f.write('\n'.join(lines))
                            number_valid += 1
                    except:
                        pass
                    
                    self.seed = np.random.random_integers(0, 10000)
                    j += 1
                
                self.seed = correct_seed
                
                current_loss = self.__loss_function(data_batch, data_predictions, z_mu, z_var)
                total_loss += current_loss[0].item()
                reconstruction_loss += current_loss[1].item()
                kl_loss += current_loss[2].item()
                predictions_complete.extend(data_predictions.cpu().numpy())
                bottleneck_complete.extend(x_sample.cpu().numpy())
                if valid:
                    valid_id.append(list_indexes[i])
                else:
                    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/molecular/run_{}/Results_not_valid.txt'.format(times), 'a') as f:
                        lines = ['Smile: {}'.format(self.ohf.back_to_smile(list(data_batch))),
                                 'Index: {}'.format(list_indexes[i]),
                                 '\n']
                        f.write('\n'.join(lines))
                
        print(number_valid)
        total_loss = total_loss / len(data_loader)
        reconstruction_loss = reconstruction_loss / len(data_loader)
        kl_loss = kl_loss / len(data_loader)

        free_memory = [data_loader, data_predictions]
        for item in free_memory:
            del item
        gc.collect()
        
        lines = ['* GET ONLY VALIDS *',
                'Number of smiles: {} '.format(len(dataset)),
                'Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(total_loss, reconstruction_loss, kl_loss),
                'Number of valid molecules: {} ({:.2f}%)'.format(number_valid, float((number_valid / len(dataset)) * 100)),
                '\n']
        create_report(self.filename_report, lines)
        
        return predictions_complete, bottleneck_complete, valid_id
    
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
            number_valid = 0
            for data_batch in data_loader:
                data_batch = data_batch.to(self.device)
                valid = False
                z_mu, z_var = model_trained.encoder(data_batch)
                correct_seed = copy.copy(self.seed)
                
                j = 0
                while j < times and not valid:
                    std = torch.exp(z_var/2)
                    eps = torch.randn_like(std) * 1e-2
                    x_sample = eps.mul(std).add_(z_mu)
                    data_predictions = model_trained.decoder(x_sample)  # output predicted by the model
                    smile_o = self.ohf.back_to_smile(list(data_predictions))
                    
                    try:
                        m = Chem.MolFromSmiles(smile_o[0])
                        if m is not None:
                            valid = True
                            number_valid += 1
                    except:
                        pass
                    
                    self.seed = np.random.random_integers(0, 10000)
                    j += 1
                
                self.seed = correct_seed
                
                if valid:
                    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/10times/Results_valid.txt', 'a') as f:
                        lines = ['Smile: {}'.format(self.ohf.back_to_smile(list(data_batch))),
                                 'Found valid smile after {} iterations'.format(j+1),
                                 'Output: {}'.format(smile_o),
                                 '\n']
                        f.write('\n'.join(lines))
                else:
                    with open('/hps/research1/icortes/acunha/python_scripts/Drug_sensitivity/data/10times/Results_not_valid.txt', 'a') as f:
                        lines = ['Smile: {}'.format(self.ohf.back_to_smile(list(data_batch))),
                                 '\n']
                        f.write('\n'.join(lines))
                

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