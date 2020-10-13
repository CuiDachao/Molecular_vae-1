# -*- coding: utf-8 -*-

# -------------------------------------------------- IMPORTS --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------- DEFINE SEEDS --------------------------------------------------

seed = 42
torch.manual_seed(seed)

# -------------------------------------------------- VARIATIONAL AUTOENCODER --------------------------------------------------

class Molecular_VAE(nn.Module):
    def __init__(self, **kwargs):
        super(Molecular_VAE, self).__init__()

        '''Defining the sizes of the different outputs
        Convolution layers
        Input: (N batch size, Cin number of channels, Lin length of signal sequence)
        Output: (N, Cout, Lout)
        where Lout = [ [ Lin + 2 * padding{in this case 0} - dilation{in this case 1} * (kernel_size - 1) - 1 ] / stride{in this case 1} ] + 1 
        '''
        self.cin = int(kwargs['number_channels_in'])
        self.lin = float(kwargs['length_signal_in'])
        self.dropout_prob = float(kwargs['dropout_prob'])
        
            
        '''Definition of the different layers'''
        self.conv1 = nn.Conv1d(self.cin, 9, kernel_size = 9)
        lout = ((self.lin + 2.0 * 0.0 - 1.0 * (9.0 - 1.0) - 1.0 ) / 1.0 ) + 1.0
        
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
        lout = ((lout + 2.0 * 0.0 - 1.0 * (9.0 - 1.0) - 1.0 ) / 1.0 ) + 1.0
        
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11)
        lout = ((lout + 2.0 * 0.0 - 1.0 * (11.0 - 1.0) - 1.0 ) / 1.0 ) + 1.0
        
        self.fc1 = nn.Linear(10 * int(lout), 425) #the input is the channels from the previous layers * lout
        self.fc2_mu = nn.Linear(425, 292)
        self.fc2_var = nn.Linear(425, 292)
        self.fc3 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first = True)
        self.fc4 = nn.Linear(501, int(self.lin))
    
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob)
        z_mu = self.fc2_mu(x)
        z_var = self.fc2_var(x)
        return z_mu, z_var
    
    def reparametrize(self, z_mu, z_var):
        if self.training:
            std = torch.exp(z_var/2)
            eps = torch.randn_like(std) * 1e-2
            x_sample = eps.mul(std).add_(z_mu)
            return x_sample
        else:
            return z_mu
    
    def decoder(self, z):
        z = F.selu(self.fc3(z))
        z = F.dropout(z, self.dropout_prob)
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.cin, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc4(out_reshape), dim = 1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y
    
    def forward(self, x):
        z_mu, z_var = self.encoder(x)
        x_sample = self.reparametrize(z_mu, z_var)
        output = self.decoder(x_sample)
        return output, x_sample, z_mu, z_var