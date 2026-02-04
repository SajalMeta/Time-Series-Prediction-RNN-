import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

def train_model(model: nn.Module, 
                X_train: torch.Tensor, 
                y_train: torch.Tensor, 
                num_epochs: int = 100, 
                learning_rate: float = 0.01,
                device: str = 'cpu') -> Tuple[nn.Module, List[float]]:
    """Trains the LSTM model.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training targets.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').
        
    Returns:
        Tuple[nn.Module, List[float]]: Trained model and history of loss values.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model and data to device
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    model.train()
    loss_history = []

    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
            
    return model, loss_history
