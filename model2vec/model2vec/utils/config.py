class Config:
    def __init__(self):
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.001
        self.log_interval = 10  # Log training loss every 10 batches
        self.checkpoint_frequency = 2  # Save checkpoint every 2 epochs
        self.checkpoint_dir = 'checkpoints'  # Directory to save checkpoints
