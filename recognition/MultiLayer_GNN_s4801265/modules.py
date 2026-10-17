
import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline for node classification.

    The MLP uses only node features and does not consider
    graph connections.
    """

    def __init__(self, input_dim=128, hidden_dim=32,
                 output_dim=4, dropout=0.5):
        """
        Initialise the MLP architecture.

        Args:
            input_dim: Number of input node features.
            hidden_dim: Number of neurons in the hidden layer.
            output_dim: Number of classification categories.
            dropout: Probability of dropping hidden activations.
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform a forward pass through the MLP.

        Args:
            x: Node features [num_nodes, input_dim].

        Returns:
            Class logits [num_nodes, output_dim].
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
