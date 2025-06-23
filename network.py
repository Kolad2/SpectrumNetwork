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


def main():
    data_path = Path('data/data_ascii')
    data = np.load('data/data_cheby_numpy/cheby_5.npy')
    params = pd.read_csv('data/data_ascii/parameters.csv')
    
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.3, random_state=42)
    data_train = data[train_idx]
    data_test = data[test_idx]
    
    params_train = params.iloc[train_idx].copy()
    params_test = params.iloc[test_idx].copy()
    
    transform = Transforms()
    x_train = transform.get_input_network(data_train, params_train)
    y_train = transform.get_output_network(params_train)
    input_size = 7
    output_size = 2
    learning_rate = 0.01
    num_epochs = 1000
    batch_size = 32
    model = MLP(input_size, 3, output_size)
    
    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()     # для регрессии
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Создание DataLoader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Обучение модели
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    test_x = transform.get_input_network(data_test, params_test)
    test_y = model(test_x)
    test_y = transform.output_network_to_numpy(test_y)
    
    params_test["test_Mo"] = test_y[:, 0]
    params_test["test_Es"] = test_y[:, 1]
    #params.to_csv('out/new_params.csv', sep=';', index=False)
    
    fig = plt.figure(figsize=(9, 5))
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2)]
    axs[0].plot(params_test["Mo"], params_test["test_Mo"], ".")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_aspect('equal')
    axs[0].set_xlabel("Истинные значения, $M_o$")
    axs[0].set_ylabel("Модельные значения, $M_o^*$")
    axs[1].plot(params_test["Es"], params_test["test_Es"], ".")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_aspect('equal')
    axs[1].set_xlabel("Истинные значения, $E_s$")
    axs[1].set_ylabel("Модельные значения, $E_s^*$")
    plt.show()
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # первый полносвязный слой
        self.relu = nn.ReLU()  # функция активации
        self.fc2 = nn.Linear(hidden_size, output_size)  # выходной слой
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Transforms:
    def __init__(self):
        self.input_mean = None
        self.input_sgm = None
        self.output_mean = None
        self.output_sgm = None
    
    def get_input_network(self, data, params):
        x = np.column_stack((data, params["R"]))
        self.input_mean = np.mean(x, 0)
        x = x - self.input_mean
        self.input_sgm = np.sqrt(np.mean(x ** 2, 0))
        x = (x / self.input_sgm)
        x = torch.tensor(x, dtype=torch.float32)
        return x
    
    def get_output_network(self, params):
        Mo = params["Mo"]
        Es = params["Es"]
        y = np.column_stack((np.log(Mo), np.log(Es)))
        self.output_mean = np.mean(y, 0)
        y = y - self.output_mean
        self.output_sgm = np.sqrt(np.mean(y ** 2, 0))
        y = (y / self.output_sgm)
        y = torch.tensor(y, dtype=torch.float32)
        return y
    
    def output_network_to_numpy(self, y):
        y = y.detach().numpy()
        print(y)
        y = y * self.output_sgm
        y = y + self.output_mean
        return np.exp(y)


if __name__ == '__main__':
    main()