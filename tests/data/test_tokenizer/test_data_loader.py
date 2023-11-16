import os

class MockTokenizer:
    def tokenize(self, text):
        return text.split()

class TestDataLoader:
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data.append(line.strip())
        return data

    def preprocess(self):
        tokenized_data = []
        for item in self.data:
            tokenized_data.append(self.tokenizer.tokenize(item))
        return tokenized_data


if __name__ == '__main__':
    # Create a dummy data file for testing
    dummy_data = ["This is a test sentence.", "Another sentence for testing."]
    data_dir = 'tests/data'
    os.makedirs(data_dir, exist_ok=True) # Ensure tests/data exists
    dummy_file_path = os.path.join(data_dir, 'test_data.txt')

    with open(dummy_file_path, 'w') as f:
        for line in dummy_data:
            f.write(line + '\n')

    # Example usage
    tokenizer = MockTokenizer()
    data_loader = TestDataLoader(dummy_file_path, tokenizer)
    preprocessed_data = data_loader.preprocess()
    print(preprocessed_data)
