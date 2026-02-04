import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def plot_loss(loss_history: list, save_path: str = 'loss_curve.png'):
    """Plots the training loss curve.
    
    Args:
        loss_history (list): List of loss values per epoch.
        save_path (str): Path to save the image.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close()

def plot_predictions(model: torch.nn.Module, 
                     X_test: torch.Tensor, 
                     y_test: torch.Tensor, 
                     scaler: MinMaxScaler, 
                     device: str = 'cpu',
                     save_path: str = 'prediction_result.png'):
    """Plots true vs predicted values.
    
    Args:
        model (nn.Module): Trained model.
        X_test (torch.Tensor): Test features.
        y_test (torch.Tensor): True test targets.
        scaler (MinMaxScaler): Scaler used for normalization (to inverse transform).
        device (str): Device to run inference on.
        save_path (str): Path to save the image.
    """
    model.eval()
    
    # Move data to device
    X_test = X_test.to(device)
    
    with torch.no_grad():
        y_pred = model(X_test)
        
    # Move back to CPU for plotting
    y_pred = y_pred.cpu().numpy()
    y_test = y_test.numpy() # Assuming y_test was kept on CPU or moved separately? Let's be safe.
    
    # Inverse transform to get actual values
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Value', color='green', alpha=0.7)
    plt.plot(y_pred_actual, label='Predicted Value', color='red', linestyle='--')
    plt.title('Time Series Prediction: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"Prediction plot saved to {save_path}")
    plt.close()
