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
import datetime
import os

from VAE_NN import Molecular_VAE
from featurizer_SMILES import OneHotFeaturizer

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/new_results/{}'.format(filename), 'a') as f:
        f.write('\n'.join(list_comments))

# -------------------------------------------------- MOLECULAR --------------------------------------------------

class Molecular():

    def __init__(self):
        self.alpha = None
        self.maximum_length = None
        self.learning_rate = None
        self.size_batch = None
        self.n_epochs = None
        self.dropout = None
        self.gamma = None
        self.step_size = None
        self.epoch_reset = None
        self.seed = None
        self.clip_value = None

        self.perc_train = None
        self.perc_val = None
        
        self.type_lr = None
        
        self.data_from = None
        self.path_data = None

        self.ohf = None
        self.device = None
        self.run_type = None
        
        self.filename_report = None
        
        self.train_indexes = []
        self.validation_indexes = []
        self.test_indexes = []
        
    # --------------------------------------------------

    def set_parameters(self, list_parameters):
        self.alpha = float(list_parameters[0])
        self.data_from = list_parameters[1]
        self.maximum_length = int(list_parameters[2])
        self.learning_rate = float(list_parameters[3])
        self.size_batch = int(list_parameters[4])
        self.n_epochs = int(list_parameters[5])
        self.perc_train = float(list_parameters[6])
        self.perc_val = float(list_parameters[7])
        self.dropout = float(list_parameters[8])
        self.gamma = float(list_parameters[9])
        self.step_size = int(list_parameters[10])
        self.seed = int(list_parameters[11])
        self.epoch_reset = int(list_parameters[12])
        self.type_lr = list_parameters[13]
        self.run_type = list_parameters[14]
        
        if self.run_type == 'resume':
            self.n_epochs += int(list_parameters[15])
        
        if self.type_lr == 'non_cyclical':
            self.epoch_reset = self.n_epochs
            self.step_size = self.n_epochs
    
        
        global seed
        if seed != self.seed:
            self.set_seed(self.seed)
        
        self.clip_value = 0.5
        self.ohf = OneHotFeaturizer()
        
        #add information to report
        if self.run_type == 'resume':
            lines = ['\n', '*** RESUME FOR MORE {} EPOCHS *** \n'.format(list_parameters[15])]
        else:
            lines = ['** REPORT - MOLECULAR **\n',
                    '* Parameters',
                    'Alpha (1.0 is without alpha): {}'.format(self.alpha),
                    'Maximum length os smiles: {}'.format(self.maximum_length),
                    'Learning rate: {} ; Size batch: {} ; Number of epochs: {} ; Dropout: {} ; Gamma: {} ;'.format(self.learning_rate, self.size_batch, self.n_epochs,self.dropout, self.gamma),
                    'Step size: {} ; Seed: {} ; Epoch to reset: {} ; Perc. of train: {}% ; Perc of validation: {}% ; Perc of test: {}% \n'.format(self.step_size, self.seed, self.epoch_reset, int(self.perc_train*100), int(self.perc_val*100), int((100 - self.perc_train - self.perc_val)*100))]
        create_report(self.filename_report, lines)
        
        if self.run_type != 'resume':
            self.save_parameters()

    # --------------------------------------------------
    
    def set_seed(self, value):
        global seed
        np.random.seed(value)
        torch.manual_seed(value)
    
    # --------------------------------------------------
    
    def load_datasets(self):
        if self.data_from == 'zinc':
            # train_set, validation_set, test_set = self.__load_zinc()
            pass
        elif self.data_from == 'prism_chembl_zinc':
            train_set, validation_set, test_set, indexes_dict = self.__load_prism_chembl_zinc()
        
        return train_set, validation_set, test_set, indexes_dict
    
    # --------------------------------------------------
    
    def __load_zinc(self):
        whole_dataset_onehot = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/zinc/zinc_final_onehot.pkl', 'rb'))
        validation_number = int(self.perc_val * len(whole_dataset_onehot))
        train_number = int(self.perc_train * len(whole_dataset_onehot))
        
        train_set, validation_set, test_set = whole_dataset_onehot[:train_number], whole_dataset_onehot[train_number:int(train_number + validation_number)], whole_dataset_onehot[int(train_number + validation_number):]
        
        del whole_dataset_onehot
        gc.collect()
        
        lines = ['\n*Datasets',
             'Training set: {}'.format(len(train_set)),
             'Validation set: {}'.format(len(validation_set)),
             'Test set: {} \n'.format(len(test_set))]
        create_report(self.filename_report, lines)
        
        return train_set, validation_set, test_set
    
    # --------------------------------------------------
    
    def __load_prism_chembl_zinc(self):
        
        whole_dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.txt', usecols = ['index', 'Dataset'], index_col = 0, nrows = 100)
        list_indexes = shuffle(list(whole_dataset.index))
        
        del whole_dataset
        gc.collect()
        
        indexes_dict = {x:{'Train':{}, 'Validation':{}, 'Test':{}} for x in ['chembl_compounds', 'chembl_approved_drugs', 'prism', 'zinc']}
        
        validation_number = int(self.perc_val * len(list_indexes))
        train_number = int(self.perc_train * len(list_indexes))
        
        self.train_indexes = list_indexes[:train_number]
        for i in range(len(self.train_indexes)):
            values = self.train_indexes[i].split('_')
            if values[0] == 'chembl':
                if values[1] == 'compounds':
                    data = '_'.join(values[:2])
                else:
                    data = '_'.join(values[:3])
            else:
                data = values[0]
            indexes_dict[data]['Train'][self.train_indexes[i]] = i
        
        self.validation_indexes = list_indexes[train_number:int(train_number + validation_number)]
        for i in range(len(self.validation_indexes)):
            values = self.train_indexes[i].split('_')
            if values[0] == 'chembl':
                if values[1] == 'compounds':
                    data = '_'.join(values[:2])
                else:
                    data = '_'.join(values[:3])
            else:
                data = values[0]
            indexes_dict[data]['Validation'][self.validation_indexes[i]] = i
            
        self.test_indexes = list_indexes[int(train_number + validation_number):]
        for i in range(len(self.test_indexes)):
            values = self.train_indexes[i].split('_')
            if values[0] == 'chembl':
                if values[1] == 'compounds':
                    data = '_'.join(values[:2])
                else:
                    data = '_'.join(values[:3])
            else:
                data = values[0]
            indexes_dict[data]['Test'][self.test_indexes[i]] = i
        
        with open('pickle/Train_indexes.txt', 'w') as f:
            f.write('\n'.join(self.train_indexes))
        with open('pickle/Validation_indexes.txt', 'w') as f:
            f.write('\n'.join(self.validation_indexes))
        with open('pickle/Test_indexes.txt', 'w') as f:
            f.write('\n'.join(self.test_indexes))
        
        lines = ['\n*Datasets',
             'Training set: {}'.format(len(self.train_indexes)),
             'Validation set: {}'.format(len(self.validation_indexes)),
             'Test set: {} \n'.format(len(self.test_indexes))]
        create_report(self.filename_report, lines)
        
        return self.train_indexes, self.validation_indexes, self.test_indexes, indexes_dict
    
    # --------------------------------------------------

    def initialize_model(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Molecular_VAE(number_channels_in=self.maximum_length, length_signal_in=len(self.ohf.get_charset()), dropout_prob = self.dropout)
        model.to(self.device)
        
        if self.run_type != 'resume':
            lines = ['\n*About the network',
                    'Runs on: {} \n'.format(self.device)]
            create_report(self.filename_report, lines)
        
        return model

    # --------------------------------------------------

    def __train_validation(self, model, train_loader, validation_loader):
        if self.run_type == 'start':
            n_epochs_not_getting_better = 0
            best_epoch = None
            results = {'total_loss_values_training' : {},
                       'reconstruction_loss_values_training' : {},
                       'kl_loss_values_training': {},
                       'total_loss_values_validation' : {},
                       'reconstruction_loss_values_validation' : {},
                       'kl_loss_values_validation' : {},
                       'learning_rates' : {},
                       'times_training' : {},
                       'times_validation' : {}}
            start_point = 0
        elif self.run_type == 'resume':
            results = pickle.load(open('pickle/Training_Validation_results.pkl', 'wb'))
            start_point = len(list(results['total_loss_values_training'].keys())) - 1
            for i in range(start_point+1):
                if i == 0:
                    best_loss = (results['total_loss_values_training'][i], results['total_loss_values_validation'][i])
                    n_epochs_not_getting_better = 0
                else:
                    loss = results['total_loss_values_validation'][i]
                    if loss < best_loss[1]:
                        best_loss = (results['total_loss_values_training'][i], results['total_loss_values_validation'][i])
                        n_epochs_not_getting_better = 0
                    else:
                        n_epochs_not_getting_better += 1
                        
            model = self.load_model(model)
            
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        best_model = copy.deepcopy(model.state_dict())  # save the best model yet with the best accuracy and lower loss value
        decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        epoch_stop = 80

        # Training and Validation
        for epoch in range(self.n_epochs):
            if (epoch + 1) % self.epoch_reset == 0 and epoch != (self.n_epochs - 1):
                print('-' * 10)
                print('Epoch: {} of {}'.format(epoch+1, self.n_epochs))
                if epoch != 0:
                    optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
                    decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

            # epoch learning rate value
            results['learning_rates'][epoch] = optimizer.state_dict()['param_groups'][0]['lr']

            # TRAINING
            start_train_time = time.time()
            train_loss_epoch, train_reconstruction_loss_epoch, train_kl_loss_epoch = self.__train__(model, optimizer, train_loader)
            end_train_model = time.time()
            results['total_loss_values_training'][epoch] = train_loss_epoch
            results['reconstruction_loss_values_training'][epoch] = train_reconstruction_loss_epoch
            results['kl_loss_values_training'][epoch] = train_kl_loss_epoch
            results['times_training'][epoch] = end_train_model - start_train_time

            # VALIDATION
            start_validation_time = time.time()
            validation_loss_epoch, validation_reconstruction_loss_epoch, validation_kl_loss_epoch = self.__eval_mode__(model, validation_loader, 'Validation', save = False)
            end_validation_time = time.time()
            results['total_loss_values_validation'][epoch] = validation_loss_epoch
            results['reconstruction_loss_values_validation'][epoch] = validation_reconstruction_loss_epoch
            results['kl_loss_values_validation'][epoch] = validation_kl_loss_epoch 
            results['times_validation'][epoch] = end_validation_time - start_validation_time

            if epoch == 0 or validation_loss_epoch < best_loss[0][0]:  # means that this model is best one yet
                best_loss = ((validation_loss_epoch, validation_reconstruction_loss_epoch, validation_kl_loss_epoch), (train_loss_epoch, train_reconstruction_loss_epoch, train_kl_loss_epoch))
                best_model = copy.deepcopy(model.state_dict())
                n_epochs_not_getting_better = 0
                pickle.dump(best_model, open('pickle/best_model_parameters.pkl', 'wb'))
            else:
                n_epochs_not_getting_better += 1   


            with open('model_values/loss_value_while_running.txt', 'a') as f:
                f.write('Epoch: {} \n'.format(epoch))
                f.write('Total training loss : {:.2f} \n'.format(train_loss_epoch))
                f.write('Reconstruction training loss : {:.2f} \n'.format(train_reconstruction_loss_epoch))
                f.write('KL training loss : {:.2f} \n'.format(train_kl_loss_epoch))
                f.write('Total validation loss : {:.2f} \n'.format(validation_loss_epoch))
                f.write('Reconstruction validation loss : {:.2f} \n'.format(validation_reconstruction_loss_epoch))
                f.write('KL validation loss : {:.2f} \n'.format(validation_kl_loss_epoch))
                f.write('Best total loss: {:.2f} \n'.format(best_loss[1][0]))
                f.write('Best reconstruction loss: {:.2f} \n'.format(best_loss[1][1]))
                f.write('Best kl loss: {:.2f} \n'.format(best_loss[1][2]))
                f.write('Time (training): {:.2f} \n'.format(end_train_model - start_train_time))
                f.write('Time (validation): {:.2f} \n'.format(end_validation_time - start_validation_time))
                f.write('Not getting better for {} epochs. \n'.format(n_epochs_not_getting_better))
                f.write('\n'.format(n_epochs_not_getting_better))

            # decay the learning rate
            decay_learning_rate.step()
            
            pickle.dump(results, open('pickle/Training_Validation_results.pkl', 'wb'))

            if n_epochs_not_getting_better == epoch_stop:
                break

        # plot the variation of the train loss, validation loss and learning rates
        results = pd.DataFrame.from_dict(results)
        results.columns = ['Total_loss_Training', 'Reconstruction_loss_Training', 'KL_loss_Training', 'Total_Loss_Validation', 'Reconstruction_loss_Validation', 'KL_loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
        results.to_csv('Training_Validation_results.txt', header=True, index=True)
        
        model.load_state_dict(best_model)
        
        print('Training: Done!')
        lines = ['\nTraining :: Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(best_loss[1][0], best_loss[1][1], best_loss[1][2]),
                'Validation :: Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(best_loss[0][0], best_loss[0][1], best_loss[0][2]),
                'Number of epochs: {:.0f} of {:.0f} \n'.format(epoch + 1, self.n_epochs)]
        create_report(self.filename_report, lines)
        
        return model

    # --------------------------------------------------
    
    def __train__(self, model, optimizer, train_loader):
        model.train()  # set model for training
        train_loss_epoch = 0.0
        train_reconstruction_loss_epoch = 0.0
        train_kl_loss_epoch = 0.0
        for train_batch in train_loader:
            train_batch = train_batch.to(self.device)
            optimizer.zero_grad()  # set the gradients of all parameters to zero
            train_predictions = model(train_batch)  # output predicted by the model
            train_current_loss = self.__loss_function(train_batch, train_predictions[0], train_predictions[2], train_predictions[3])
            train_current_loss[0].backward()  # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            optimizer.step()
            train_loss_epoch += train_current_loss[0].item()
            train_reconstruction_loss_epoch += train_current_loss[1].item()
            train_kl_loss_epoch += train_current_loss[2].item()

        train_loss_epoch = train_loss_epoch / len(train_loader)
        train_reconstruction_loss_epoch = train_reconstruction_loss_epoch / len(train_loader)
        train_kl_loss_epoch = train_kl_loss_epoch / len(train_loader)
        
        return train_loss_epoch, train_reconstruction_loss_epoch, train_kl_loss_epoch
    
    # --------------------------------------------------
    
    def __eval_mode__(self, model, data_loader, type_dataset, save = True):
        model.eval()
        loss_epoch = 0.0
        reconstruction_loss_epoch = 0.0
        kl_loss_epoch = 0.0
        predictions_complete, bottleneck_complete = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                predictions = model(batch)  # output predicted by the model
                current_loss = self.__loss_function(batch, predictions[0], predictions[2], predictions[3])
                loss_epoch += current_loss[0].item()
                reconstruction_loss_epoch += current_loss[1].item()
                kl_loss_epoch += current_loss[2].item()
                if save:
                    predictions_complete.extend(predictions[0].cpu().numpy().tolist())
                    bottleneck_complete.extend(predictions[1].cpu().numpy().tolist())
        loss_epoch = loss_epoch / len(data_loader)
        reconstruction_loss_epoch = reconstruction_loss_epoch / len(data_loader)
        kl_loss_epoch = kl_loss_epoch / len(data_loader)
        
        if save:
            pickle.dump([predictions_complete, bottleneck_complete], open('pickle/{}_outputs_bottlenecks.pkl'.format(type_dataset), 'wb'))
        
        return loss_epoch, reconstruction_loss_epoch, kl_loss_epoch

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output, z_mu, z_var):
        reconstruction_loss = F.binary_cross_entropy(x_output, x_input, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        return reconstruction_loss + (self.alpha * kl_loss), reconstruction_loss, kl_loss

    # --------------------------------------------------

    def train_model(self, model, train_set, validation_set):
        start_training = time.time()
        train_loader = [torch.tensor(train_set[i:i+self.size_batch]).type('torch.FloatTensor') for i in range(0, len(train_set), self.size_batch)]
        validation_loader = [torch.tensor(validation_set[i:i+self.size_batch]).type('torch.FloatTensor') for i in range(0, len(validation_set), self.size_batch)]

        model = self.__train_validation(model, train_loader, validation_loader)
        end_training = time.time()
        create_report(self.filename_report, ['Duration: {:.2f} \n'.format(end_training - start_training)])
        
        _, _, _ = self.__eval_mode__(model, train_loader, 'Train')
        _, _, _ = self.__eval_mode__(model, validation_loader, 'Validation')
        
        
        self.__save_model(model)

        return model

    # --------------------------------------------------

    def __save_model(self, model):
        model_parameters = copy.deepcopy(model.state_dict())
        pickle.dump(model_parameters, open('pickle/molecular_model.pkl', 'wb'))
    
    # --------------------------------------------------
    
    def load_model(self, model):
        model_parameters = pickle.load(open('pickle/molecular_model.pkl', 'rb'))
        model.load_state_dict(best_model)
        return model
    
    # --------------------------------------------------
        
    def save_parameters(self):
        pickle.dump(
            [np.format_float_positional(self.alpha), self.data_from, self.maximum_length, np.format_float_positional(self.learning_rate),
             self.size_batch, self.n_epochs, self.perc_train, self.perc_val, self.dropout, self.gamma,
             self.step_size, self.seed, self.epoch_reset], open('pickle/list_initial_parameters_smiles.pkl', 'wb'))
    
    # --------------------------------------------------

    def run_test_set(self, model, test_set):
        test_loader = [torch.tensor(test_set[i:i+self.size_batch]).type('torch.FloatTensor') for i in range(0, len(test_set), self.size_batch)]
        test_loss, test_reconstruction_loss, test_kl_loss = self.__eval_mode__(model, test_loader, 'Test')
        
        del test_loader
        gc.collect()

        create_report(self.filename_report, ['Testing :: Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(test_loss, test_reconstruction_loss, test_kl_loss)])

    # --------------------------------------------------
    
    def count_valid(self, type_dataset):
        whole_dataset = pd.read_csv('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/prism_chembl250_chembldrugs_zinc250.txt', usecols = ['index', 'Smile'], index_col = 0)

        valid = 0
        same = 0
        
        if type_dataset == 'Train':
            indexes = self.train_indexes
        elif type_dataset == 'Validation':
            indexes = self.validation_indexes
        else:
            indexes = self.test_indexes
        
        smiles_i = list(whole_dataset['Smile'].loc[indexes])
        
        onehot_o = pickle.load(open('pickle/{}_outputs_bottlenecks.pkl'.format(type_dataset), 'rb'))
        smiles_o = self.ohf.back_to_smile(onehot_o[0])
        
        with open('{}_smiles_predictions.txt'.format(type_dataset), 'w') as f:
            for i in range(len(smiles_i)):
                s_i = smiles_i[i]
                s_o = smiles_o[i]
                m = Chem.MolFromSmiles(s_o)
                if m is not None:
                    valid += 1
                if s_i == s_o:
                    same += 1
                f.write('\n'.join(['Input: {}'.format(s_i), 'Output: {}'.format(s_o), '\n']))
                f.write('\n')
        
        create_report(self.filename_report, ['\n', 'For {}_set: '.format(type_dataset), 'Valid molecules: {:.2f}%'.format((valid/len(smiles_i)) * 100),
                                             'Same as input: {}'.format(same)])
        
    # --------------------------------------------------

    def plot_loss_lr(self, x, loss_training, loss_validation, kl_training, kl_validation, recons_training, recons_validation, learning_rates):
        fig = plt.figure(figsize=(12, 16))
        (ax1, ax3) = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(x, loss_training, '-r', label='Total Loss (training)')
        ax1.set_ylabel('Loss')
        ax1.plot(x, loss_validation, '-g', label = 'Total Loss (validation)')
        ax1.plot(x, kl_training, ':r', label='KL Loss (training)')
        ax1.plot(x, kl_validation, ':g', label = 'KL Loss (validation)')
        ax1.plot(x, recons_training, '-.r', label='Reconstruction Loss (training)')
        ax1.plot(x, recons_validation, '-.g', label = 'Reconstruction Loss (validation)')
        ax3.set_xlabel('Number of epochs')
        ax3.set_ylabel('Learning rates')
        ax3.plot(x, learning_rates, color='b', label = 'Learning rates')
        fig.legend(loc=1)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('plots/Loss_learningRate_per_epoch.png', bbox_inches='tight')
        
    # --------------------------------------------------

    def create_filename(self, list_parameters):
        filename_output = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(list_parameters[2], list_parameters[3], list_parameters[4],
                                                                 list_parameters[5], list_parameters[6], list_parameters[7],
                                                                 list_parameters[8], list_parameters[9], list_parameters[10],
                                                                 list_parameters[11], list_parameters[12], list_parameters[13])
        self.filename_report = 'alpha_{}/{}/output_{}.txt'.format(list_parameters[0], list_parameters[1], filename_output)
        return self.filename_report

# -------------------------------------------------- RUN --------------------------------------------------

def run_molecular(list_parameters, run_type):
    start_run = time.time()
    print(str(datetime.datetime.now().time()))
    molecular = Molecular()
    
    if run_type == 'resume':
        more_epoch = list_parameters[-1]
        list_parameters = pickle.load(open('pickle/list_initial_parameters_smiles.pkl', 'rb'))
        filename = molecular.create_filename(list_parameters)
        list_parameters.extend([run_type, more_epoch])
    else:
        molecular.create_filename(list_parameters)
    
    molecular.set_parameters(list_parameters)
    
    train_set_indexes, validation_set_indexes, test_set_indexes, indexes_dict = molecular.load_datasets() 
    
    datasets_from = ['chembl_compounds', 'chembl_approved_drugs', 'prism', 'zinc']
    onehot_dict = {x:{} for x in ['Train', 'Validation', 'Test']}
    for data in datasets_from:
        path = '/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/onehot_{}'.format(data, data) 
        files = os.listdir(path)
        for file in files:
            whole_dataset_onehot = pickle.load(open('/hps/research1/icortes/acunha/python_scripts/Molecular_vae/data/PRISM_ChEMBL_ZINC/onehot_{}/{}'.format(data, file), 'rb'))
            onehot_indexes = list(whole_dataset_onehot.keys())
            
            train_i = list(set(onehot_indexes).intersection(list(indexes_dict[data]['Train'].keys())))
            validation_i = list(set(onehot_indexes).intersection(list(indexes_dict[data]['Validation'].keys())))
            test_i = list(set(onehot_indexes).intersection(list(indexes_dict[data]['Test'].keys())))
            
            for index in train_i:
                i = indexes_dict[data]['Train'][index]
                onehot_dict['Train'][i] = whole_dataset_onehot[index]
            
            for index in validation_i:
                i = indexes_dict[data]['Validation'][index]
                onehot_dict['Validation'][i] = whole_dataset_onehot[index]
            
            for index in test_i:
                i = indexes_dict[data]['Test'][index]
                onehot_dict['Test'][i] = whole_dataset_onehot[index]
    
    train_set = []
    validation_set = []
    test_set = []
    for k in sorted(onehot_dict['Train'].keys()):
        train_set.append(onehot_dict['Train'][k])
    for k in sorted(onehot_dict['Validation'].keys()):
        validation_set.append(onehot_dict['Validation'][k])
    for k in sorted(onehot_dict['Test'].keys()):
        test_set.append(onehot_dict['Test'][k])
    
    del onehot_dict
    gc.collect()
    
    model = molecular.initialize_model()
    model_trained = molecular.train_model(model, train_set, validation_set)
    
    free_memory = [train_set, validation_set]
    for item in free_memory:
        del item
    gc.collect()
        
    molecular.run_test_set(model_trained, test_set)
    
    del test_set
    gc.collect()
    
    #count the valid smiles predictions
    type_dataset = ['Train', 'Validation', 'Test']
    for type_d in type_dataset:
        molecular.count_valid(type_d)
    
    #plots
    results = pd.read_csv('Training_Validation_results.txt', header = 0, index_col = 0)
    molecular.plot_loss_lr(list(list(results.index)), list(results['Total_loss_Training']), list(results['Total_Loss_Validation']),
                           list(results['KL_loss_Training']), list(results['KL_loss_Validation']),
                           list(results['Reconstruction_loss_Training']), list(results['Reconstruction_loss_Validation']),
                           list(results['Learning_rates']))
    
    results_barplots = results.loc[results.index % 10 == 0]
    results_barplots.loc[:, ['Duration_Training', 'Duration_Validation']].plot(kind='bar', rot=0, subplots=True, figsize=(16, 8))
    plt.savefig('plots/Duration_per_epoch.png', bbox_inches='tight')
    
    print('Done!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    input_values = sys.argv[1:]
    run_type = input_values[-1]
    run_molecular(input_values, run_type)

except EOFError:
    print('ERROR!')