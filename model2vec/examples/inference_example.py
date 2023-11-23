import torch
from model2vec.models.simple_model import SimpleModel
from model2vec.inference.embedder import generate_embedding

# Example Usage
if __name__ == '__main__':
    # Instantiate the model
    model = SimpleModel()

    # Load a pre-trained model
    model.load_state_dict(torch.load('trained_model.pth'))  # Replace 'trained_model.pth' with the actual path to your trained model

    # Create some dummy input data
    input_data = torch.randn(1, 10)

    # Generate the embedding
    embedding = generate_embedding(model, input_data)

    # Print the embedding
    print("Generated Embedding:", embedding)
