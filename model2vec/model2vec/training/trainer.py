import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, data_loader, optimizer, loss_fn, device='cpu'):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(self.device)

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.data_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')
