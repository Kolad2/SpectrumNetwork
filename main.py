import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from numpy.polynomial import chebyshev

def main():
    data_path = Path('data/data_ascii')
    params = pd.read_csv('data/data_ascii/parameters.csv')
    datas = []
    cheby_c = []
    for index, row in params.iterrows():
        path = (data_path / "spectrum" / row["FileName"]).with_suffix('.txt')
        data = np.loadtxt(path, skiprows=1)
        datas.append(data)
        freq = data[:, 0]
        freq = 2*(freq - freq.min())/(freq.max() - freq.min())-1
        dens = data[:, 1]
        cheby_c.append(chebyshev.chebfit(freq, dens, deg=16))
    cheby_c = np.array(cheby_c)
    np.save('data/data_cheby_numpy/cheby_16.npy', cheby_c)
    

    plt.figure(figsize=(10, 5))
    for data, c in zip(datas, cheby_c):
        freq = data[:, 0]
        freq = 2*(freq - freq.min()) / (freq.max() - freq.min())-1
        dens = data[:, 1]
        cheby = chebyshev.Chebyshev(c)
        y = cheby(freq)
        plt.plot(freq, dens, 'b-', linewidth=1)
        plt.plot(freq, y, 'b-', linewidth=1)
        # Настройка оформления
        plt.title('Спектр: D(F)')
        plt.xlabel('Frequency (F)')
        plt.ylabel('Amplitude (D)')
        plt.grid(True, linestyle='--', alpha=0.7)
        #plt.legend()
        
        # Сохранение и отображение
        plt.tight_layout()
        plt.savefig('spectrum_plot.png', dpi=300)  # Сохраняем график
        plt.show()


if __name__ == '__main__':
    main()