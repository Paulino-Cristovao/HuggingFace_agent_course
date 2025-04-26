# training.py

"""
Contrail Model Training Module

This script trains a neural network model based on ResNet18 and tabular atmospheric
data to predict aircraft contrail formation probability. It saves the trained model
and the feature scaler for later use in inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18
import joblib

class ContrailPredictor(nn.Module):
    """
    Neural network model using pretrained ResNet18 to predict contrail formation from tabular atmospheric data.
    """

    def __init__(self, input_features: int):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512 + input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, image_tensor: torch.Tensor, tabular_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            image_tensor (torch.Tensor): Dummy image input tensor.
            tabular_tensor (torch.Tensor): Scaled tabular features tensor.

        Returns:
            torch.Tensor: Probability of contrail formation.
        """
        image_features = self.feature_extractor(image_tensor)
        combined_features = torch.cat([image_features, tabular_tensor], dim=1)
        output = self.classifier(combined_features)

        return torch.tensor(output) if not isinstance(output, torch.Tensor) else output


def load_data(path: str) -> pd.DataFrame:
    """
    Load contrail tabular data from a CSV file.

    Args:
        path (str): File path to the CSV data.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(path)


def train_model(
    df: pd.DataFrame,
    epochs: int = 200,
    model_path: str = "contrail_model.pth",
    scaler_path: str = "scaler.pkl"
) -> None:
    """
    Train the ContrailPredictor model and save the model and scaler.

    Args:
        df (pd.DataFrame): DataFrame containing atmospheric data and contrail labels.
        epochs (int): Number of training epochs.
        model_path (str): Path to save the trained model.
        scaler_path (str): Path to save the trained scaler.

    Returns:
        None
    """
    X: np.ndarray = df.drop('contrail', axis=1).values
    y: np.ndarray = df['contrail'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)

    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ContrailPredictor(input_features=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    dummy_image_tensor = torch.zeros((X_train_tensor.size(0), 3, 224, 224)).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(dummy_image_tensor, X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    data_path = "data/fake_contrail_data.csv"
    df = load_data(data_path)
    train_model(df)
# training.py