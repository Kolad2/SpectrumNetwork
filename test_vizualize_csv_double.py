import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from numpy.polynomial import chebyshev


def main():
    dates = [20190706, 20201024, 20221112, 20231118, 20231216]
    test_dates = [20190706, 20231216]
    model_name = "_".join(str(date) for date in test_dates)
    params = pd.read_csv("outputs/data_test_" + model_name + ".csv")
    
    params1 = params.iloc[params.index[params["Date"] != test_dates[0]]]
    params2 = params.iloc[params.index[params["Date"] != test_dates[1]]]
    
    fig = plt.figure(figsize=(8, 8))
    axs = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4)
    ]
    
    def plot_mo_es_errors(ax1, ax2, params, name):
        Mo = params["Mo"].to_numpy()
        Es = params["Es"].to_numpy()
        test_Mo = params["test_Mo"].to_numpy()
        test_Es = params["test_Es"].to_numpy()
        
        Mo_error = np.abs(Mo - test_Mo) / Mo
        Es_error = np.abs(Es - test_Es) / Es
        
        ax1.hist(Mo_error, bins=40, color='blue', alpha=0.7)
        ax1.set_title("$M_o$" + " " + name)
        ax1.set_xlabel('Относительная ошибка')
        ax1.set_ylabel('Частота')
        
        # Гистограмма для Es_error
        ax2.hist(Es_error, bins=40, color='red', alpha=0.7)
        ax2.set_title("$E_s$" + " " + name)
        ax2.set_xlabel('Относительная ошибка')
        ax2.set_ylabel('Частота')
    
    plot_mo_es_errors(axs[0], axs[1], params1, str(test_dates[0]))
    plot_mo_es_errors(axs[2], axs[3], params2, str(test_dates[1]))
    plt.tight_layout()
    fig.savefig("./outputs/pictures/" + "pic" + model_name + ".png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()