import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib.lines import lineStyles
from numpy.polynomial import chebyshev
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from models import MLP


def main():
    data_path = Path('data/data_ascii')
    data = np.load('data/data_cheby_numpy/cheby_8.npy')
    params = pd.read_csv('data/data_ascii/parameters.csv')
    config = {
        "input_size": 7,
        "learning_rate": 0.0001,
        "num_epochs": 5000,
        "batch_size": 32,
        "hidden_size": 30
    }
    dates = [20190706, 20201024, 20221112, 20231118, 20231216]
    date = dates[3]
    unique_dates = params["Date"].unique()
    
    # train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.3, random_state=42)
    train_idx = params.index[params["Date"] != date]
    test_idx = params.index[params["Date"] == date]
    
    data_train = data[train_idx]
    params_train = params.iloc[train_idx].copy()

    data_test = data[test_idx]
    params_test = params.iloc[test_idx].copy()
    
    trainer = MLPTrainer(data_train, params_train, data_test, params_test, config)
    trainer.train()
    trainer.save("./saved_models/model_" + str(date) + ".pth")
    #trainer.load("./saved_models/model_" + str(date) + ".pth")
    trainer.test_evaluate()
    trainer.test_save("./outputs/data_test_" + str(date) + ".csv")
    
    vizualize(trainer.params_test)
    

class MLPTrainer:
    def __init__(self, data_train, params_train, data_test, params_test, config):
        self.data_train = data_train
        self.params_train = params_train
        self.data_test = data_test
        self.params_test = params_test
        self.config = config
        self.transform = Transforms()
        self.transform.configurate(self.data_train, self.params_train)
        self.model: Optional[MLP] = None
    
    def train(self):
        x_train = self.transform.get_input_network(self.data_train, self.params_train)
        y_train = self.transform.get_output_network(self.params_train)
        self.model = MLP(x_train.size()[1], self.config["hidden_size"], 2)
        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        train(self.model, dataloader, self.config["num_epochs"], self.config["learning_rate"])
    
    def test_evaluate(self):
        test_x = self.transform.get_input_network(self.data_test, self.params_test)
        test_y = self.model(test_x)
        test_y = self.transform.output_network_to_numpy(test_y)
        self.params_test["test_Mo"] = test_y[:, 0]
        self.params_test["test_Es"] = test_y[:, 1]
    
    def test_save(self, path):
        self.params_test.to_csv(path, index=False, encoding='utf-8')
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = MLP.load(path)
        
        
def vizualize(params_test):
    fig = plt.figure(figsize=(12, 5))
    axs = [fig.add_subplot(1, 2, 1),
           fig.add_subplot(1, 2, 2)]
    axs[0].plot(params_test["Mo"], params_test["test_Mo"], ".")
    mo_min = np.min(params_test["Mo"])
    mo_max = np.max(params_test["Mo"])
    min_max = np.array([mo_min, mo_max])
    axs[0].plot(min_max, min_max * 0.7, linestyle="-", color="black")
    axs[0].plot(min_max, min_max * 1.3, linestyle="-", color="black")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_aspect('equal')
    axs[0].set_xlabel("Истинные значения, $M_o$")
    axs[0].set_ylabel("Модельные значения, $M_o^*$")
    axs[1].plot(params_test["Es"], params_test["test_Es"], ".")
    es_min = np.min(params_test["Es"])
    es_max = np.max(params_test["Es"])
    min_max = np.array([es_min, es_max])
    axs[1].plot(min_max, min_max * 0.7, linestyle="-", color="black")
    axs[1].plot(min_max, min_max * 1.3, linestyle="-", color="black")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_aspect('equal')
    axs[1].set_xlabel("Истинные значения, $E_s$")
    axs[1].set_ylabel("Модельные значения, $E_s^*$")
    plt.show()


def train(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # для регрессии

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


class Transforms:
    def __init__(self):
        self.input_mean = None
        self.input_sgm = None
        self.output_mean = None
        self.output_sgm = None
    
    def configurate(self, data, params):
        x = np.column_stack((data, params["R"]))
        self.input_mean = np.mean(x, 0)
        x = x - self.input_mean
        self.input_sgm = np.sqrt(np.mean(x ** 2, 0))
        y = np.column_stack((np.log(params["Mo"]), np.log(params["Es"])))
        self.output_mean = np.mean(y, 0)
        self.output_mean = np.mean(y, 0)
        y = y - self.output_mean
        self.output_sgm = np.sqrt(np.mean(y ** 2, 0))
        
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
        y = y * self.output_sgm
        y = y + self.output_mean
        return np.exp(y)


if __name__ == '__main__':
    main()
    

# # params.to_csv('out/new_params.csv', sep=';', index=False)
        # rmse = np.sqrt(np.mean(
        #     (np.log(params_test["test_Mo"]) - np.log(params_test["Mo"])) ** 2
        # ))
        #
        # params_test.to_csv('./outputs/data_test.csv', index=False, encoding='utf-8')
        # params_train.to_csv('./outputs/data_train.csv', index=False, encoding='utf-8')
        #
        # vizualize(params_test)
        # return rmse, model