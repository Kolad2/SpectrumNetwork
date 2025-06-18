import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from numpy.polynomial import chebyshev


def main():
    data_path = Path('data/data_ascii')
    params = pd.read_csv('data/data_ascii/parameters.csv')
    data = np.load('data/data_cheby_numpy/cheby_5.npy')
    
    x = np.linspace(-1, 1, 10)
    
    plt.figure(figsize=(10, 5))
    for c in data:
        cheby = chebyshev.Chebyshev(c)
        y = cheby(x)
        plt.plot(x, y, 'b-', linewidth=1)
        # Настройка оформления
        plt.title('Спектр: D(F)')
        plt.xlabel('Frequency (F)')
        plt.ylabel('Amplitude (D)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Сохранение и отображение
        plt.tight_layout()
        plt.savefig('spectrum_plot.png', dpi=300)  # Сохраняем график
        plt.show()


if __name__ == '__main__':
    main()