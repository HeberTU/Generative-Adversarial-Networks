# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:45:02 2020

@author: HTRUJILLO
"""


import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, output_size)
        
        
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        
        x = x.view(-1, 28*28)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        
        out = self.fc4(x)
        
        return out


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        
        self.fc4 = nn.Linear(hidden_dim*4, output_size)

        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        
        out = F.tanh(self.fc4(x))
        
        return out