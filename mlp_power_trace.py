import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, traces, labels):
        self.traces = traces
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        trace = torch.Tensor(self.traces[idx])
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        return {
            'traces': trace,
            'labels': label
        }

    

class PowerTraceMLP(nn.Module):
    def __init__(self):
        super(PowerTraceMLP, self).__init__()
        self.fc1 = nn.Linear(400, 20)
        self.hidden_activation = nn.ReLU()
        self.fc2 = nn.Linear(20, 256)

    def forward(self, x):
        x = self.hidden_activation(self.fc1(x))
        x = self.fc2(x)
        return x