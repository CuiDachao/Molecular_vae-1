# -*- coding: utf-8 -*-

# -------------------------------------------------- IMPORTS --------------------------------------------------

import numpy as np
import pickle
import torch

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------------------------- ONE HOT ENCODING --------------------------------------------------

class OneHotFeaturizer():

    def __init__(self):
        self.charset = [' ', '#', '(', ')', '+', '-', '/',
                        '1', '2', '3', '4', '5', '6', '7',
                        '8', '9', '=', '@', 'B', 'C', 'F', 'H',
                        'I', 'N', 'O', 'P', 'S', '[', '\\',
                        ']', 'c', 'l', 'n', 'o', 'r', 's']
        
        self.charset_index = {}

    def featurize(self, smiles, max_legth):
        # self.charset = self._create_charset(smiles)
        self._create_charset_index()
        maximum_length = max_legth
        
        one_hot_encoder_list = []
        for smile in smiles:
            try:
                X = np.zeros((maximum_length, len(self.charset)))
                for i in range(len(smile)):
                    c_index = self.charset_index[smile[i]]
                    X[i, c_index] = 1
                if len(smile) < maximum_length:
                    for i in range(len(smile), maximum_length):
                        c_index = self.charset_index[' ']
                        X[i, c_index] = 1
            except KeyError:
                X = float('NaN')
            one_hot_encoder_list.append(X)
        return one_hot_encoder_list

    def back_to_smile(self, list_one_hot):
        smile_list = []
        for X in list_one_hot:
            smile = ''
            try:
                indexes = np.argmax(X.type(torch.FloatTensor).detach().numpy(), axis = 1) #if the input is a tensor
            except:
                indexes = np.argmax(X, axis = 1)
            for i in indexes:
                smile += self.charset[i]
            smile_list.append(smile.strip())
        return smile_list
    
    def _create_charset_index(self):
        for i in range(len(self.charset)):
            self.charset_index[self.charset[i]] = i

    # def _create_charset(self, smiles):
    #     charset = set()
    #     for smile in smiles:
    #         for c in smile:
    #             charset.add(c)
    #     charset.add(' ')
    #     return sorted(list(charset))
    
    def get_charset(self):
        return self.charset