import torch
from torch import nn


class Model(nn.Module):
    """Simple neural network classifier for the Iris dataset."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    model = Model()

    x = torch.rand(1, 4)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape of model: {output.shape}")