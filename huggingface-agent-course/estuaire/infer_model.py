# inference.py

"""
Contrail Prediction and Recommendation Module

This module predicts aircraft contrail formation probability
based on atmospheric conditions and generates flight recommendations
to minimize climate impact using AI (OpenAI GPT models).
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torchvision.models import resnet18
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from typing import Optional

from openai import OpenAI, OpenAIError

# Load environment variables (OpenAI API Key)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in the environment or in a .env file.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


class ContrailPredictor(torch.nn.Module):
    """
    Neural network model using pretrained ResNet18 to predict contrail formation from tabular atmospheric data.
    """

    def __init__(self, input_features: int):
        super().__init__()
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.fc = torch.nn.Identity()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 + input_features, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
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
        # Ensure output is a tensor
        return output if isinstance(output, torch.Tensor) else torch.tensor(output)


def predict_contrail(model_path: str, scaler_path: str, input_data: NDArray[np.float64]) -> float:
    """
    Predict contrail formation probability based on atmospheric data.

    Args:
        model_path (str): Path to the saved PyTorch model weights.
        scaler_path (str): Path to the saved scikit-learn StandardScaler.
        input_data (np.ndarray): Atmospheric parameters (altitude, temperature, humidity, speed).

    Returns:
        float: Probability of contrail formation (0.0 to 1.0).
    """
    scaler: StandardScaler = joblib.load(scaler_path)
    input_scaled = scaler.transform(input_data.reshape(1, -1))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ContrailPredictor(input_features=input_data.shape[0]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tabular_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    dummy_image_tensor = torch.zeros((1, 3, 224, 224)).to(device)

    with torch.no_grad():
        probability = model(dummy_image_tensor, tabular_tensor).item()

    return probability


def generate_recommendation(probability: float, model: str = "gpt-4") -> str:
    """
    Generate a pilot recommendation based on contrail formation probability.

    Args:
        probability (float): Predicted probability of contrail formation (0.0 to 1.0).
        model (str): OpenAI model name ("gpt-4" or "gpt-3.5-turbo").

    Returns:
        str: AI-generated recommendation for pilots.

    Raises:
        OpenAIError: If the OpenAI API call fails.
    """
    prompt = (
        f"Given a predicted contrail formation probability of {probability:.0%}, "
        "provide a concise recommendation to pilots on altitude adjustments to minimize climate impact."
    )

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an aviation expert helping pilots minimize contrails."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=100            # Environment variables
            .env
            .env.*
            
            # Model files
            *.pth
            *.pkl
            *.pt
            *.bin
            *.onnx
            
            # Data directories and files
            data/
            dataset/
            datasets/
            *.csv
            *.tsv
            *.json
            *.jsonl
            *.parquet
            *.npy
            *.npz
            
            # Python cache
            __pycache__/
            *.py[cod]
            *$py.class
            .ipynb_checkpoints/
            *.so
            .Python
            build/
            develop-eggs/
            dist/
            downloads/
            eggs/
            .eggs/
            lib/
            lib64/
            parts/
            sdist/
            var/
            wheels/
            *.egg-info/
            .installed.cfg
            *.egg
            
            # IDE specific files
            .idea/
            .vscode/
            *.swp
            *.swo
            .DS_Store
            
            # Logs
            logs/
            *.log
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e


if __name__ == "__main__":
    # Example atmospheric data: altitude (ft), temperature (°C), humidity (%), speed (knots)
    sample_input = np.array([35000, -55, 80, 480])

    try:
        contrail_prob = predict_contrail(
            model_path='contrail_model.pth',
            scaler_path='scaler.pkl',
            input_data=sample_input
        )

        recommendation = generate_recommendation(contrail_prob)

        print(f"Contrail Probability: {contrail_prob:.2%}")
        print("Recommendation:", recommendation)

    except Exception as e:
        print(f"Error: {e}")
