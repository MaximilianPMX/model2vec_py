import torch
from model2vec.models.simple_model import SimpleModel # Assuming simple_model.py is in the models directory

def generate_embedding(model, input_data):
    model.eval()
    with torch.no_grad():
        embedding = model.fc1(input_data) # Example: Using output of the first fully connected layer as embedding
    return embedding.numpy()

if __name__ == '__main__':
    model = SimpleModel()
    # Load a trained model (replace 'path/to/trained_model.pth' with your actual path)
    # model.load_state_dict(torch.load('path/to/trained_model.pth'))

    # Dummy Input
    dummy_input = torch.randn(1, 10)  # Example of input data for the model
    embedding = generate_embedding(model, dummy_input)

    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")