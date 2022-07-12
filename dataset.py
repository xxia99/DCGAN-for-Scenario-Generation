import torch
from torch.utils.data import Dataset
import pandas
import os

class MyDataset(Dataset):
    def __init__(self, root):
        self.electricity = torch.tensor(
            pandas.read_csv(os.path.join(root, 'Electricity.csv'))['MaxElectricity'].to_numpy()[:17520].reshape(365, 48, 1), dtype=torch.float)
        self.gas = torch.tensor(
            pandas.read_csv(os.path.join(root, 'Gas.csv'))['Gas'].to_numpy()[:17520].reshape(365, 48, 1), dtype=torch.float)
        self.solar = torch.tensor(
            pandas.read_csv(os.path.join(root, 'Solar.csv'))['Solar'].to_numpy()[:17520].reshape(365, 48, 1), dtype=torch.float)
        self.wind = torch.tensor(
            pandas.read_csv(os.path.join(root, 'Wind.csv'))['Wind'].to_numpy()[:17520].reshape(365, 48, 1), dtype=torch.float)
        #print(self.electricity.min(), self.electricity.max()) #14.8490 151.7346
        #print(self.gas.min(), self.gas.max()) #30 30
        #print(self.solar.min(), self.solar.max()) #0 29.2381
        #print(self.wind.min(), self.wind.max()) #0 111.2749
        self.electricity = (self.electricity - 90.) / 90.  # (min, max)=(0, 180) normalize to (-1, 1)
        self.gas = (self.gas - 30.) / 30.  # (min, max)=(30, 30) normalize to (-1, 1)
        self.solar = (self.solar - 15.) / 15.  # (min, max)=(0, 30) normalize to (-1, 1)
        self.wind = (self.wind - 60.) / 60.  # (min, max)=(0, 120) normalize to (-1, 1)
        self.data = torch.cat([self.electricity, self.gas, self.solar, self.wind], dim=2)
        self.data = self.data.reshape(365, 1, 48, 4)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

#data shape: 48, 4
def plot(data, save_path=None, show=True):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 48, 48)
    electricity = data[:, 0] * 90 + 90
    gas = data[:, 1] * 30 + 30
    solar = data[:, 2] * 15 + 15
    wind = data[:, 3] * 20 + 20

    plt.plot(x, electricity, label='electricity', c = (1, 0, 0), marker='o')
    plt.plot(x, gas, label='gas', c = (0, 1, 0), marker='o')
    plt.plot(x, solar, label='solar', c=(0, 0, 1), marker='o')
    plt.plot(x, wind, label='wind', c=(1, 1, 0), marker='o')
    plt.legend(loc='lower left')

    plt.title('Intergrated Energy Scenario')
    #plt.xlim(0, 48)
    #plt.ylim(0, 200)
    plt.xlabel('Time(30min)')
    plt.ylabel('Power(MW)')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    plt.close()

#plot(MyDataset('./Datasets')[2][0])