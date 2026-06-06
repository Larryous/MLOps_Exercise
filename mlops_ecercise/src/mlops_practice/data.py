from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):
    """Small custom PyTorch dataset using the Iris dataset."""

    def __init__(self, data_path: Path | str, split: str = "train") -> None:
        """
        Args:
            data_path: Path to the processed data folder.
            split: Either "train" or "test".
        """

        self.data_path = Path(data_path)
        self.split = split

        x_file = self.data_path / f"X_{split}.pt"
        y_file = self.data_path / f"y_{split}.pt"

        if not x_file.exists() or not y_file.exists():
            raise FileNotFoundError(
                f"Processed files not found in {self.data_path}. "
                f"Run preprocessing first."
            )

        self.X = torch.load(x_file)
        self.y = torch.load(y_file)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.X[index], self.y[index]

    @staticmethod
    def preprocess(output_folder: Path | str) -> None:
        """
        Preprocess the Iris data and save it to the output folder.

        The Iris dataset is small and built into scikit-learn.
        We standardize the features and save train/test tensors.
        """

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        iris = load_iris()

        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        torch.save(X_train, output_folder / "X_train.pt")
        torch.save(X_test, output_folder / "X_test.pt")
        torch.save(y_train, output_folder / "y_train.pt")
        torch.save(y_test, output_folder / "y_test.pt")

        print(f"Saved processed data to: {output_folder}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")


def preprocess(
    data_path: Path = Path("data/raw"),
    output_folder: Path = Path("data/processed"),
) -> None:
    """
    Preprocess data command.

    data_path is included to match the project structure,
    but the Iris dataset is loaded directly from scikit-learn.
    """

    print("Preprocessing data...")
    print(f"Raw data path argument: {data_path}")
    MyDataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)