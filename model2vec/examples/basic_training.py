import torch
import torch.optim as optim
from model2vec.models.simple_model import SimpleModel
from model2vec.data.data_loader import create_dataloader, DummyDataset
from model2vec.training.trainer import train

# Example Usage
if __name__ == '__main__':
    # Instantiate the model
    model = SimpleModel()

    # Create a dummy dataset and dataloader
    dataset = DummyDataset()
    dataloader = create_dataloader(dataset, batch_size=32)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, dataloader, optimizer, epochs=5)

    # Save the trained model (optional)
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Training completed and model saved to trained_model.pth")