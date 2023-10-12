# model2vec

## Project Description

The model2vec project focuses on generating static embeddings for models. It includes modules for training, inference, and distillation, with supporting documentation, tests, and tutorials. This plan outlines the recreation of the project, prioritizing core functionality and incremental development.

## Basic Usage

### Training

To train the simple linear model, run the following command:

```bash
python model2vec/examples/basic_training.py
```

This will train a simple linear model and save the trained model. You can modify the training parameters in the `model2vec/examples/basic_training.py` file.

### Inference

To generate embeddings from a trained model, run the following command:

```bash
python model2vec/model2vec/inference/embed_from_file.py --model_path path/to/your/trained_model.pth --output_path path/to/your/embeddings.npy
```

Replace `path/to/your/trained_model.pth` with the actual path to your trained model file and `path/to/your/embeddings.npy` with the desired path for the output embeddings file.

### Running Unit Tests

To run the unit tests, use the following command:

```bash
pytest
```

This will discover and run all the tests in the `tests` directory.
