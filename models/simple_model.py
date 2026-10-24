
import torch.nn as nn


class SimpleDensityModel(nn.Module):
    """
    Small model for the federated-learning demonstration.

    Input:
        2 numerical features:
        1. Normalized crowd count
        2. Normalized crowd density

    Output:
        1 value between 0 and 1
    """

    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
