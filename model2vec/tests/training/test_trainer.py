import unittest
import torch
from model2vec.models.simple_model import SimpleModel
from model2vec.data.data_loader import create_dataloader, DummyDataset
from model2vec.training.trainer import train

class TestTrainer(unittest.TestCase):

    def test_trainer(self):
        model = SimpleModel()
        dataset = DummyDataset(size=100)
        data_loader = create_dataloader(dataset, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train(model, data_loader, optimizer, epochs=1)
        # Assert that the model's parameters have changed during training
        for param in model.parameters():
            self.assertFalse(torch.all(param.grad == 0))

if __name__ == '__main__':
    unittest.main()