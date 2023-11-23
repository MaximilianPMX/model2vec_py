import torch
import torch.optim as optim
from model2vec.models.simple_model import SimpleModel # Assuming simple_model.py is in the models directory
from model2vec.data.data_loader import create_dataloader, DummyDataset # Assuming data_loader.py is in the data directory

def train(model, data_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9: # Print every 10 batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    model = SimpleModel()
    dataset = DummyDataset()
    data_loader = create_dataloader(dataset, batch_size=32)
    optimizer = optim.Adam(model.parameters())
    train(model, data_loader, optimizer)
