import torch
import os
import json
from model2vec.inference.embedder import ModelEmbedder
from model2vec.utils.config import Config


def embed_from_file(
    config_path: str,
    model_path: str,
    input_file: str,
    output_file: str,
    device: str = "cpu",
):
    """Loads a model, reads input from a file, generates embeddings, and saves them to a file.

    Args:
        config_path (str): Path to the configuration file.
        model_path (str): Path to the trained model.
        input_file (str): Path to the input text file.
        output_file (str): Path to save the embeddings.
        device (str): Device to use for inference (default: "cpu").
    """
    config = Config.from_json(config_path)

    embedder = ModelEmbedder(config, model_path, device=device)

    with open(input_file, "r") as f:
        input_text = f.read().strip()

    embedding = embedder.embed(input_text)

    # Convert embedding to a list for JSON serialization
    embedding_list = embedding.tolist()
    
    with open(output_file, "w") as f:
        json.dump(embedding_list, f)

    print(f"Embedding saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    config_path = "config.json"  # Replace with your actual config path
    model_path = "model.pth"  # Replace with your actual model path
    input_file = "input.txt"  # Replace with your input text file
    output_file = "embedding.json"  # Replace with your desired output file

    # Create dummy config.json and input.txt if they don't exist
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({"embedding_dim": 10, "model_type": "simple"}, f) # minimal config

    if not os.path.exists(input_file):
        with open(input_file, "w") as f:
            f.write("This is an example input.")
    
    embed_from_file(config_path, model_path, input_file, output_file)
