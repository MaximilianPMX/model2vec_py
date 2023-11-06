import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

    def train(self, data_loader):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(data_loader):
                inputs = data['input'].to(self.device)
                labels = data['label'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Log training information
                if (i + 1) % self.config.log_interval == 0:
                    avg_loss = running_loss / self.config.log_interval
                    logging.info(f'Epoch [{epoch+1}/{self.config.num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {avg_loss:.4f}')
                    running_loss = 0.0

            # Checkpoint the model
            if (epoch + 1) % self.config.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(self.config.checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
                logging.info(f'Checkpoint saved to {checkpoint_path}')

        logging.info('Finished Training')
