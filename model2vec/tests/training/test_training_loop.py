import unittest
from unittest.mock import MagicMock
import torch
from model2vec.training.trainer import Trainer


class TestTrainingLoop(unittest.TestCase):

    def test_training_loop_runs_without_error(self):
        # Mock data loader
        mock_data_loader = MagicMock()
        mock_data_loader.__len__.return_value = 10  # Mock the length
        mock_data_loader.__iter__.return_value = [((torch.randn(1, 10), torch.randint(0, 2, (1,))),
                                                    (torch.randn(1, 10), torch.randint(0, 2, (1,)))) for _ in range(10)]  # Mock the iterator to return batches

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 2)  # Mock the forward pass
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]  # Mock the parameters

        # Mock optimizer
        mock_optimizer = MagicMock()

        # Mock loss function
        mock_loss_function = MagicMock(return_value=torch.tensor(0.5))  # Mock some loss 

        # Instantiate trainer with mocks
        trainer = Trainer(mock_model, mock_data_loader, mock_data_loader, mock_optimizer, loss_function=mock_loss_function)

        # Run training loop for a small number of epochs
        try:
            trainer.train(epochs=2)
        except Exception as e:
            self.fail(f"Training loop raised an exception: {e}")

    def test_training_loop_loss_decreases(self):
        # Mock data loader
        mock_data_loader = MagicMock()
        mock_data_loader.__len__.return_value = 10
        mock_data_loader.__iter__.return_value = [((torch.randn(1, 10), torch.randint(0, 2, (1,))),
                                                    (torch.randn(1, 10), torch.randint(0, 2, (1,)))) for _ in range(10)]

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 2)
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]

        # Mock optimizer
        mock_optimizer = MagicMock()

        # Mock loss function, track loss values
        loss_values = [torch.tensor(1.0), torch.tensor(0.8), torch.tensor(0.6), torch.tensor(0.4), torch.tensor(0.2)]
        loss_index = 0

        def mock_loss(*args, **kwargs):
            nonlocal loss_index
            loss = loss_values[loss_index % len(loss_values)]
            loss_index += 1
            return loss

        mock_loss_function = MagicMock(side_effect=mock_loss)

        # Instantiate trainer with mocks
        trainer = Trainer(mock_model, mock_data_loader, mock_data_loader, mock_optimizer, loss_function=mock_loss_function)

        # Run training loop and check if loss decreases
        initial_loss = mock_loss_function()
        trainer.train(epochs=1)
        final_loss = mock_loss_function()

        self.assertLessEqual(final_loss.item(), initial_loss.item(), "Loss should decrease during training.")


if __name__ == '__main__':
    unittest.main()