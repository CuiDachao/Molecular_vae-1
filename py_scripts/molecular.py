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

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- FUNCTIONS --------------------------------------------------

def create_report(filename, list_comments):
    with open("/hps/research1/icortes/acunha/python_scripts/Molecular_vae/results/{}".format(filename), 'a') as f:
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

        self.ohf = None
        self.device = None
        
        self.filename_report = None
        
    # --------------------------------------------------

    def set_parameters(self, list_parameters):
        self.alpha = float(list_parameters[0])
        self.maximum_length = int(list_parameters[1])
        self.learning_rate = float(list_parameters[2])
        self.size_batch = int(list_parameters[3])
        self.n_epochs = int(list_parameters[4])
        self.perc_train = float(list_parameters[5])
        self.perc_val = float(list_parameters[6])
        self.dropout = float(list_parameters[7])
        self.gamma = float(list_parameters[8])
        self.step_size = int(list_parameters[9])
        self.seed = int(list_parameters[10])
        self.epoch_reset = int(list_parameters[11])
        self.clip_value = 0.5
        self.ohf = OneHotFeaturizer()
        
        #add information to report
        lines = ["** REPORT - MOLECULAR **\n",
                "* Parameters",
                "Alpha (1.0 is without alpha): {}".format(self.alpha),
                "Maximum length os smiles: {}".format(self.maximum_length),
                "Learning rate: {} ; Size batch: {} ; Number of epochs: {} ; Dropout: {} ; Gamma: {} ;".format(self.learning_rate, self.size_batch, self.n_epochs,self.dropout, self.gamma),
                "Step size: {} ; Seed: {} ; Epoch to reset: {} ; Perc. of train: {}% ; Perc of validation: {}% ; Perc of test: {}% \n".format(self.step_size, self.seed, self.epoch_reset, int(self.perc_train*100), int(self.perc_val*100), int((100 - self.perc_train - self.perc_val)*100))]
        create_report(self.filename_report, lines)

        global seed
        if seed != self.seed:
            seed = self.seed
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    # --------------------------------------------------

    def load_datasets(self):
        with open('/hps/research1/icortes/acunha/data/ZINC_PRISM_SMILES/zinc_prism_smiles_processed.smi') as f:
            whole_dataset = []
            for smi in f:
                whole_dataset.append(smi.strip("\n"))
        
        whole_dataset = check_valid_smiles(whole_dataset, self.maximum_length)
        print('Valid smiles: {}'.format(len(whole_dataset)))
        
        whole_dataset = self.ohf.featurize(whole_dataset, self.maximum_length)
        whole_dataset = [x for x in whole_dataset if str(x) != 'nan']
        print("One hot encoded matrices: {}".format(len(whole_dataset)))

        
        # Split the dataset
        validation_number = int(self.perc_val * len(whole_dataset))
        train_number = int(self.perc_train * len(whole_dataset))

        train_set, validation_set, test_set = whole_dataset[:train_number], whole_dataset[train_number:int(train_number + validation_number)], whole_dataset[int(train_number + validation_number):]
        
        del whole_dataset
        gc.collect()
        
        pickle.dump(train_set, open('pickle/train_set.pkl', 'wb'))
        pickle.dump(validation_set, open('pickle/validation_set.pkl', 'wb'))
        pickle.dump(test_set, open('pickle/test_set.pkl', 'wb'))
        
        lines = ["\n*Datasets",
             "Training set: {}".format(len(train_set)),
             "Validation set: {}".format(len(validation_set)),
             "Test set: {} \n".format(len(test_set))]
        create_report(self.filename_report, lines)
        
        return train_set, validation_set, test_set

# --------------------------------------------------

    def initialize_model(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        model = Molecular_VAE(number_channels_in=self.maximum_length,
                              length_signal_in=len(self.ohf.get_charset()), dropout_prob = self.dropout)
        model.to(self.device)
        
        #save parameters as a pkl file
        self.save_parameters()
        
        lines = ["\n*About the network",
                "Runs on: {} \n".format(self.device)]
        create_report(self.filename_report, lines)
        
        return model

    # --------------------------------------------------

    def __train_validation(self, model, train_set, validation_set):
        # Divide the training dataset into batches
        train_set_torch = torch.tensor(train_set).type('torch.FloatTensor')
        validation_set_torch = torch.tensor(validation_set).type('torch.FloatTensor')
        
        train_loader = torch.utils.data.DataLoader(train_set_torch, batch_size=self.size_batch, shuffle=False)
        validation_loader = torch.utils.data.DataLoader(validation_set_torch, batch_size=self.size_batch, shuffle=False)
        
        free_memory = [train_set_torch, validation_set_torch]
        for item in free_memory:
            del item
        gc.collect()
        
        epoch_stop = int(2.3 * self.epoch_reset)
        got_better = False
        n_epochs_not_getting_better = 0

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        best_model = copy.deepcopy(model.state_dict())  # save the best model yet with the best accuracy and lower loss value
        decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # Save the loss values (plot) + learning rates of each epoch (plot)
        total_loss_values_training = {}  # different loss values in different epochs (training)
        reconstruction_loss_values_training = {}  # different reconstruction loss values in different epochs (training)
        kl_loss_values_training = {}  # different kl loss values in different epochs (training)
        total_loss_values_validation = {}  # different loss values in different epochs (validation)
        reconstruction_loss_values_validation = {}  # different reconstruction loss values in different epochs (validation)
        kl_loss_values_validation = {}  # different kl loss values in different epochs (validation)
        learning_rates = {}  # learning rates values per epoch
        times_training = {}  # time spent per epoch (training)
        times_validation = {}  # time spent per epoch (validation)

        # Training and Validation
        for epoch in range(self.n_epochs):
            train_loss_epoch = 0.0
            train_reconstruction_loss_epoch = 0.0
            train_kl_loss_epoch = 0.0
            validation_loss_epoch = 0.0
            validation_reconstruction_loss_epoch = 0.0
            validation_kl_loss_epoch = 0.0

            if (epoch + 1) % self.epoch_reset == 0 and epoch != (self.n_epochs - 1):
                print('-' * 10)
                print('Epoch: {} of {}'.format(epoch+1, self.n_epochs))
                if epoch != 0:
                    optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
                    decay_learning_rate = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

            # epoch learning rate value
            learning_rates[epoch] = optimizer.state_dict()['param_groups'][0]['lr']

            # TRAINING
            train_predictions_complete, train_bottleneck_complete = [], []
            start_train_time = time.time()
            model.train()  # set model for training
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
                train_predictions_complete.extend(list(train_predictions[0].detach().cpu().numpy()))
                train_bottleneck_complete.extend(list(train_predictions[1].detach().cpu().numpy()))

            end_train_model = time.time()

            # epoch training values
            train_loss_epoch = train_loss_epoch / len(train_loader)
            train_reconstruction_loss_epoch = train_reconstruction_loss_epoch / len(train_loader)
            train_kl_loss_epoch = train_kl_loss_epoch / len(train_loader)
            total_loss_values_training[epoch] = train_loss_epoch
            reconstruction_loss_values_training[epoch] = train_reconstruction_loss_epoch
            kl_loss_values_training[epoch] = train_kl_loss_epoch
            times_training[epoch] = end_train_model - start_train_time

            # VALIDATION
            start_validation_time = time.time()
            model.eval()
            validation_predictions_complete, validation_bottleneck_complete = [], []
            with torch.no_grad():
                for validation_batch in validation_loader:
                    validation_batch = validation_batch.to(self.device)
                    optimizer.zero_grad()  # set the gradients of all parameters to zero
                    validation_predictions = model(validation_batch)  # output predicted by the model
                    validation_current_loss = self.__loss_function(validation_batch, validation_predictions[0], validation_predictions[2], validation_predictions[3])
                    validation_loss_epoch += validation_current_loss[0].item()
                    validation_reconstruction_loss_epoch += validation_current_loss[1].item()
                    validation_kl_loss_epoch += validation_current_loss[2].item()
                    validation_predictions_complete.extend(list(validation_predictions[0].cpu().numpy()))
                    validation_bottleneck_complete.extend(list(validation_predictions[1].cpu().numpy()))

            end_validation_time = time.time()

            # epoch validation values
            validation_loss_epoch = validation_loss_epoch / len(validation_loader)
            validation_reconstruction_loss_epoch = validation_reconstruction_loss_epoch / len(validation_loader)
            validation_kl_loss_epoch = validation_kl_loss_epoch / len(validation_loader)
            total_loss_values_validation[epoch] = validation_loss_epoch
            reconstruction_loss_values_validation[epoch] = validation_reconstruction_loss_epoch
            kl_loss_values_validation[epoch] = validation_kl_loss_epoch 
            times_validation[epoch] = end_validation_time - start_validation_time

            if epoch == 0 or validation_loss_epoch < best_loss[0][0]:  # means that this model is best one yet
                best_loss = ((validation_loss_epoch, validation_reconstruction_loss_epoch, validation_kl_loss_epoch), (train_loss_epoch, train_reconstruction_loss_epoch, train_kl_loss_epoch))
                best_model = copy.deepcopy(model.state_dict())
                pickle.dump([validation_predictions_complete, validation_bottleneck_complete], open('pickle/validation_outputs_bottlenecks.pkl', 'wb'), protocol = 4)
                pickle.dump([train_predictions_complete, train_bottleneck_complete], open('pickle/train_outputs_bottlenecks.pkl', 'wb'), protocol=4)
                got_better = True
                n_epochs_not_getting_better = 0
                
                pickle.dump(best_model, open('pickle/best_model_parameters.pkl', 'wb'))

                
                with open("model_values/validation_smiles.txt", 'w') as f:
                        smiles = self.ohf.back_to_smile(validation_predictions_complete)
                        smiles.insert(0, 'Epoch {} (best model): \n'.format(epoch))
                        f.write('\n'.join(smiles))
            else:
                got_better = False
                n_epochs_not_getting_better += 1
            
            if (epoch + 1) % 200 == 0:
                model_parameters = copy.deepcopy(model.state_dict())
                pickle.dump(model_parameters, open('pickle/model_parameters_{}.pkl'.format(epoch), 'wb'))
                
            
            free_memory = [validation_bottleneck_complete, validation_predictions_complete, train_bottleneck_complete, train_predictions_complete]
            for item in free_memory:
                del item
            gc.collect()

            with open("model_values/loss_value_while_running.txt", 'a') as f:
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

            if n_epochs_not_getting_better == epoch_stop:
                break

            if (epoch + 1) % 50 == 0:
                results = pd.DataFrame.from_dict((total_loss_values_training, reconstruction_loss_values_training, kl_loss_values_training, total_loss_values_validation, reconstruction_loss_values_validation, kl_loss_values_validation, learning_rates, times_training, times_validation)).T
                results.columns = ['Total_loss_Training', 'Reconstruction_loss_Training', 'KL_loss_Training', 'Total_Loss_Validation', 'Reconstruction_loss_Validation', 'KL_loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
                results.reset_index().to_csv('Training_Validation_results.txt', header=True, index=False)
                
                del results
                gc.collect()

        free_memory = [train_loader, validation_loader]
        for item in free_memory:
            del item
        gc.collect()


        # plot the variation of the train loss, validation loss and learning rates
        results = pd.DataFrame.from_dict((total_loss_values_training, reconstruction_loss_values_training, kl_loss_values_training, total_loss_values_validation, reconstruction_loss_values_validation, kl_loss_values_validation, learning_rates, times_training, times_validation)).T
        results.columns = ['Total_loss_Training', 'Reconstruction_loss_Training', 'KL_loss_Training', 'Total_Loss_Validation', 'Reconstruction_loss_Validation', 'KL_loss_Validation', 'Learning_rates', 'Duration_Training', 'Duration_Validation']
        results.reset_index().to_csv('Training_Validation_results.txt', header=True, index=False)
        
        model.load_state_dict(best_model)
        print('Training: Done!')
        lines = ['\nTraining :: Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(best_loss[1][0], best_loss[1][1], best_loss[1][2]),
                'Validation :: Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(best_loss[0][0], best_loss[0][1], best_loss[0][2]),
                'Number of epochs: {:.0f} of {:.0f} \n'.format(epoch + 1, self.n_epochs)]
        create_report(self.filename_report, lines)
        
        return model

    # --------------------------------------------------

    def __loss_function(self, x_input, x_output, z_mu, z_var):
        reconstruction_loss = F.binary_cross_entropy(x_output, x_input, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu.pow(2) - 1.0 - z_var)
        return reconstruction_loss + (self.alpha * kl_loss), reconstruction_loss, kl_loss

    # --------------------------------------------------

    def train_model(self, model, train_set, validation_set):
        start_training = time.time()
        model = self.__train_validation(model, train_set, validation_set)
        end_training = time.time()
        create_report(self.filename_report, ['Duration: {:.2f} \n'.format(end_training - start_training)])
        self.__save_model(model)

        return model

    # --------------------------------------------------

    def __save_model(self, model):
        model_parameters = copy.deepcopy(model.state_dict())
        pickle.dump(model_parameters, open('pickle/molecular_model.pkl', 'wb'))
    
    # --------------------------------------------------
        
    def save_parameters(self):
        pickle.dump(
            [self.alpha, self.maximum_length, self.learning_rate, self.size_batch, self.n_epochs, self.perc_train,
             self.perc_val, self.dropout, self.gamma, self.step_size, self.seed, self.epoch_reset, self.device],
            open('pickle/list_initial_parameters_smiles.pkl', 'wb'))
    
    # --------------------------------------------------

    def run_test_set(self, model, test_set):
        test_set_torch = torch.tensor(test_set).type('torch.FloatTensor')
        test_loader = torch.utils.data.DataLoader(test_set_torch, batch_size=self.size_batch, shuffle=False)
        
        test_loss = 0.0
        test_reconstruction_loss = 0.0
        test_kl_loss = 0.0
        test_predictions_complete, test_bottleneck_complete = [], []
        model.eval()
        with torch.no_grad():
            for test_batch in test_loader:
                test_batch = test_batch.to(self.device)
                test_predictions = model(test_batch)  # output predicted by the model
                current_loss = self.__loss_function(test_batch, test_predictions[0], test_predictions[2], test_predictions[3])
                test_loss += current_loss[0].item()
                test_reconstruction_loss += current_loss[1].item()
                test_kl_loss += current_loss[2].item()
                test_predictions_complete.extend(list(test_predictions[0].cpu().numpy()))
                test_bottleneck_complete.extend(list(test_predictions[1].cpu().numpy()))

        test_loss = test_loss / len(test_loader)
        test_reconstruction_loss = test_reconstruction_loss / len(test_loader)
        test_kl_loss = test_kl_loss / len(test_loader)
        
        pickle.dump([test_predictions_complete, test_bottleneck_complete], open('pickle/test_outputs_bottlenecks.pkl', 'wb'))
        
        free_memory = [test_loader, test_bottleneck_complete]
        for item in free_memory:
            del item
        gc.collect()
        
        valid = 0
        with open("test_smiles.txt", 'w') as f:
            smiles_i = self.ohf.back_to_smile(test_set)
            smiles_o = self.ohf.back_to_smile(test_predictions_complete)
            for i in range(len(smiles_i)):
                s_i = smiles_i[i]
                s_o = smiles_o[i]
                m = Chem.MolFromSmiles(s_o)
                if m is not None:
                    valid += 1
                f.write('\n'.join(["Input: {}".format(smiles_i[i]), "Output: {}".format(smiles_o[i]), '\n']))
                f.write('\n')
        
        free_memory = [test_set_torch, test_predictions_complete]
        for item in free_memory:
            del item
        gc.collect()
                
        print('Valid molecules: {:.2f}%'.format((valid/len(smiles_o)) * 100))
        create_report(self.filename_report, ['Testing :: Total loss: {:.2f} ; Reconstruction loss: {:.2f} ; KL loss: {:.2f}'.format(test_loss, test_reconstruction_loss, test_kl_loss),
                               'Valid molecules: {:.2f}%'.format((valid/len(smiles_o)) * 100)])

    # --------------------------------------------------

    def plot_loss_lr(self, x, loss_training, loss_validation, kl_training, kl_validation, recons_training, recons_validation, learning_rates):
        
        minimum = min(min(loss_training), min(loss_validation), min(kl_training), min(kl_validation), min(recons_training), min(recons_validation))
        maximum = max(max(loss_training), max(loss_validation), max(kl_training), max(kl_validation), max(recons_training), max(recons_validation))
        
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
        filename_output = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(list_parameters[1], list_parameters[2], list_parameters[3], list_parameters[4], list_parameters[5], list_parameters[6], list_parameters[7], list_parameters[8], list_parameters[9], list_parameters[11])
        self.filename_report = "{}/output_{}.txt".format(list_parameters[0], filename_output)

# -------------------------------------------------- RUN --------------------------------------------------

def run_molecular(list_parameters):
    molecular = Molecular()
    molecular.create_filename(list_parameters)
    molecular.set_parameters(list_parameters)
    
    train_set, validation_set, test_set = molecular.load_datasets()
    
    model = molecular.initialize_model()
    model_trained = molecular.train_model(model, train_set, validation_set)
    
    free_memory = [train_set, validation_set]
    for item in free_memory:
        del item
    gc.collect()
        
    molecular.run_test_set(model_trained, test_set)
    
    del test_set
    gc.collect()
    
    #plots
    results = pd.read_csv("Training_Validation_results.txt", header = 0, index_col = 0)
    molecular.plot_loss_lr(list(list(results.index)), list(results['Total_loss_Training']), list(results['Total_Loss_Validation']),
                           list(results['KL_loss_Training']), list(results['KL_loss_Validation']),
                           list(results['Reconstruction_loss_Training']), list(results['Reconstruction_loss_Validation']),
                           list(results['Learning_rates']))
    
    results_barplots = results.loc[results.index % 10 == 0]
    results_barplots.loc[:, ['Duration_Training', 'Duration_Validation']].plot(kind="bar", rot=0, subplots=True, figsize=(16, 8))
    plt.savefig('plots/Duration_per_epoch.png', bbox_inches='tight')
    
    print('Done!')

# -------------------------------------------------- INPUT --------------------------------------------------

try:
    input_values = sys.argv[1:]
    run_molecular(input_values)

except EOFError:
    print('ERROR!')