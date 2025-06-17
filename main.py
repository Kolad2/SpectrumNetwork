import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def main():
    data_path = Path('data_ascii')
    params = pd.read_csv('data_ascii/parameters.csv')
    datas = []
    for index, row in params.iterrows():
        path = (data_path / "spectrum" / row["FileName"]).with_suffix('.txt')
        data = np.loadtxt(path, skiprows=1)
        datas.append(data)
    
    data = datas[0]
    
    return
    # Построение графика
    plt.figure(figsize=(10, 5))
    for data in datas:
        # Разделение на F и D
        freq = data[:, 0]  # Первая колонка - F (частота/ось X)
        d = data[:, 1]  # Вторая колонка - D (амплитуда/ось Y)
        plt.plot(freq, d, 'b-', linewidth=1)
    
    # Настройка оформления
    plt.title('Спектр: D(F)')
    plt.xlabel('Frequency (F)')
    plt.ylabel('Amplitude (D)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Сохранение и отображение
    plt.tight_layout()
    plt.savefig('spectrum_plot.png', dpi=300)  # Сохраняем график
    plt.show()


if __name__ == '__main__':
    main()