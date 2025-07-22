import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from numpy.polynomial import chebyshev


def main():
    dates = [20190706, 20201024, 20221112, 20231118, 20231216]
    date = dates[4]
    params = pd.read_csv("outputs/data_test_" + str(date) + ".csv")
    
    Mo = params["Mo"].to_numpy()
    Es = params["Es"].to_numpy()
    test_Mo = params["test_Mo"].to_numpy()
    test_Es = params["test_Es"].to_numpy()
    
    Mo_error = np.abs(Mo - test_Mo) / Mo
    Es_error = np.abs(Es - test_Es) / Es
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Гистограмма для Mo_error
    ax1.hist(Mo_error, bins=40, color='blue', alpha=0.7)
    ax1.set_title("$M_o$" + " " + str(date))
    ax1.set_xlabel('Относительная ошибка')
    ax1.set_ylabel('Частота')
    
    # Гистограмма для Es_error
    ax2.hist(Es_error, bins=40, color='red', alpha=0.7)
    ax2.set_title("$E_s$" + " " + str(date))
    ax2.set_xlabel('Относительная ошибка')
    ax2.set_ylabel('Частота')
    
    plt.tight_layout()
    fig.savefig("./outputs/pictures/" + str(date) + ".png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()