from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mlops_practice.model import Model
from mlops_practice.data import MyDataset

def train() -> None:
    """Train the model and save it to the models folder."""

    processed_data_path = Path("data/processed")
    model_output_path = Path("models")
    model_output_path.mkdir(parents=True, exist_ok=True)

    if not processed_data_path.exists():
        print("Processed data not found. Creating processed data...")
        MyDataset.preprocess(processed_data_path)

    train_dataset = MyDataset(processed_data_path, split="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )

    model = Model()

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 50

    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")

    print("Training finished.")

    model_path = model_output_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train()
