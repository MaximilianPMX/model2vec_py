import torch
from model2vec.utils.config import Config

class Embedder:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def load_model(self):
        # Load the model based on the configuration.
        # This part needs to be adapted based on your specific model loading method.
        # Example: Assuming a simple model loading from a saved file.
        model_path = self.config.model_path
        self.model = torch.load(model_path)
        self.model.eval()

    def generate_embedding(self, input_data) -> torch.Tensor:
        """Generates an embedding for the given input data.

        Args:
            input_data: The input data to be fed into the model.  This should be
                         appropriately preprocessed based on the model's requirements.

        Returns:
            A torch.Tensor representing the embedding.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Ensure input_data is a torch.Tensor and move to the correct device
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data) # Assuming input_data can be converted to tensor

        input_data = input_data.to(self.config.device)  # Move data to device

        with torch.no_grad(): # Disable gradient calculation during inference.
            # Forward pass through the model.
            output = self.model(input_data)

            # Extract the embedding from the desired layer's output.
            # This needs to be configured based on your model architecture and the
            # specific layer you want to use for the embedding.  For example, if you want
            # the output from the last layer, you might use:
            embedding = output  # Or output[layer_index] if extracting from a specific layer

            # If you need to extract a specific layer output you can also use named_modules() or a hook.
            # For example:
            # for name, module in self.model.named_modules():
            #     if name == 'your_layer_name':
            #       embedding = module(input_data)
            #       break

            return embedding
