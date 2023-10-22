import unittest
import numpy as np
from model2vec.data.data_loader import DataLoader  # Replace with actual path

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a dummy data file for testing
        self.dummy_data = ["This is a test sentence.", "Another sentence for testing."]
        self.data_file = "test_data.txt"
        with open(self.data_file, "w") as f:
            for line in self.dummy_data:
                f.write(line + "\n")

    def tearDown(self):
        # Remove the dummy data file
        import os
        os.remove(self.data_file)

    def test_data_loading(self):
        # Test that data is loaded correctly from the file
        loader = DataLoader(self.data_file)
        loaded_data = loader.load_data()
        self.assertEqual(len(loaded_data), len(self.dummy_data))
        for i in range(len(self.dummy_data)):
            self.assertEqual(loaded_data[i], self.dummy_data[i])

    def test_split_data(self):
        loader = DataLoader(self.data_file)
        loaded_data = loader.load_data()
        train_data, val_data = loader.split_data(loaded_data, split_ratio=0.8)
        self.assertEqual(len(train_data), 1)
        self.assertEqual(len(val_data), 1)


    def test_preprocess_data(self):
        # Test the preprocessing step (tokenization, etc.)
        # Adapt this test according to the actual preprocessing in DataLoader
        loader = DataLoader(self.data_file)
        loaded_data = loader.load_data()
        preprocessed_data = loader.preprocess_data(loaded_data)
        self.assertIsInstance(preprocessed_data, list)
        # Add assertions to check the content of preprocessed_data
        # For example if your tokenizer lowercases:
        self.assertEqual(preprocessed_data[0], self.dummy_data[0])


if __name__ == '__main__':
    unittest.main()