from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mlops_practice.data import MyDataset
from mlops_practice.model import Model

def evaluate() -> None:
    """Load a trained model and evaluate it on the test set."""

    processed_data_path = Path("data/processed")
    model_path = Path("models/model.pt")

    if not processed_data_path.exists():
        raise FileNotFoundError(
            f"Processed data folder not found: {processed_data_path}. "
            "Run preprocessing first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run training first."
        )

    test_dataset = MyDataset(processed_data_path, split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            predicted_classes = torch.argmax(predictions, dim=1)

            correct += (predicted_classes == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total

    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate()