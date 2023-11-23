import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    model = SimpleModel()
    dummy_input = torch.randn(1, 10) # Example input size
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")