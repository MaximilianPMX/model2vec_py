import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,)) # Binary labels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    dummy_dataset = DummyDataset()
    data_loader = create_dataloader(dummy_dataset)
    for i, (data, labels) in enumerate(data_loader):
        print(f'Batch {i+1}: Data shape: {data.shape}, Labels shape: {labels.shape}')
        break
