import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from numpy.polynomial import chebyshev

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)  # первый полносвязный слой
        self.relu = nn.ReLU()  # функция активации
        self.fc2 = nn.Linear(hidden_size, output_size)  # выходной слой

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def save(self, path):
        torch.save({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'state_dict': self.state_dict(),
        }, path)
        
    @classmethod
    def load(cls, path):
        """Загружает модель, восстанавливая архитектуру"""
        checkpoint = torch.load(path)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
    