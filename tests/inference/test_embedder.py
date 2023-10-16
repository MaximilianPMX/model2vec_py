import pytest
import torch
from model2vec.inference.embedder import Embedder
from model2vec.models.simple_model import SimpleModel  # Import a concrete model class for testing


@pytest.fixture
def simple_model():
    # Create a simple model instance for testing
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    return model


@pytest.fixture
def embedder(simple_model):
    # Create an Embedder instance with the
    return Embedder(model=simple_model, device='cpu')


def test_embedder_initialization(embedder, simple_model):
    assert embedder.model == simple_model
    assert embedder.device == torch.device('cpu')



def test_embed_single_input(embedder):
    input_tensor = torch.randn(1, 10)
    embedding = embedder.embed(input_tensor)
    assert embedding is not None
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, 5), f"Expected (1, 5), got {embedding.shape}"



def test_embed_batch_input(embedder):
    input_tensor = torch.randn(4, 10)  # Batch size of 4
    embedding = embedder.embed(input_tensor)
    assert embedding is not None
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (4, 5), f"Expected (4, 5), got {embedding.shape}"



def test_embed_returns_float(embedder):
    input_tensor = torch.randn(1, 10)
    embedding = embedder.embed(input_tensor)
    assert embedding.dtype == torch.float32


