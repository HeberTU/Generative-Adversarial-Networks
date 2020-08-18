# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:25:42 2020

@author: HTRUJILLO
"""
import torch
from torchvision import datasets
import torchvision.transforms as transforms


def get_train_loader(num_workers = 0, batch_size = 64)

    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root = 'data', train = True,
                            download = True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           num_workers = num_workers)
    
    return train_loader