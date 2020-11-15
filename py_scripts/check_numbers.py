# -------------------------------------------------- IMPORTS --------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
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
import datetime
import os
import h5py
import random

from VAE_NN import Molecular_VAE
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/{}'.format(filename), 'a') as f:
        f.write('\n'.join(list_comments))
        
def eval_mode(model, dataset, indexes, device):    
    model.eval()
    run_n_times_max = 1000
    
    n_valid = 0
    n_same = 0
    
    res = {}
    
    with torch.no_grad():
        for i in range(len(indexes)):
            batch = torch.Tensor(data = dataset['one_hot_matrices'][sorted(indexes[i])]).type('torch.FloatTensor').to(device)
            j = 0
            valid = False
            same = False
            while j < run_n_times_max or not valid:
                predictions = model(batch)  # output predicted by the model
                s_o = ohf.back_to_smile([predictions[0].cpu().numpy()])[0]
                m = Chem.MolFromSmiles(s_o)
                if m is not None:
                    valid = True
                    n_valid += 1
                j += 1
                
            if valid:
                s_i = np.char.decode(dataset['smiles'][indexes[i]]).tolist()
                if s_i == s_o:
                    n_same += 1
                    same = True
            
            res[dataset['index'][indexes[i]]] = {'Valid' : valid, 'Same' : same, 'N_epochs' : j + 1}
                    
    return res, n_valid, n_same

# -------------------------------------------------- RUN --------------------------------------------------
ohf = OneHotFeaturizer()
filename = 'check_the_numbers.txt'

#import parameters frm best model
alpha, _, maximum_length, learning_rate, _, _, _, _, dropout, _, _, seed, _, _, _ = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model/list_initial_parameters_smiles.pkl', 'rb'))
alpha = float(alpha)
maximum_length = int(maximum_length)
learning_rate = float(learning_rate)
dropout = float(dropout)
seed = int(seed)

#define the seeds
np.random.seed(seed)
torch.manual_seed(seed)

#initialise and load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Molecular_VAE(number_channels_in=maximum_length, length_signal_in=len(ohf.get_charset()), dropout_prob = dropout)
model.to(device)
model_parameters = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/best_model/molecular_model.pkl', 'rb'))
model.load_state_dict(best_model)

#run the dataset

