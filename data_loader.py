import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Union

def generate_sine_wave_data(seq_length: int = 1000) -> np.ndarray:
    """Generates synthetic sine wave data for testing.
    
    Args:
        seq_length (int): Total number of data points to generate.
        
    Returns:
        np.ndarray: A 1D array containing the sine wave values.
    """
    x = np.linspace(0, 100, seq_length)
    y = np.sin(x)
    return y

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sliding window sequences for LSTM training.
    
    Args:
        data (np.ndarray): The normalized time series data.
        seq_length (int): The length of the input sequence (lookback window).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X: Input sequences of shape (num_samples, seq_length).
            - y: Target values of shape (num_samples,).
    """
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

def prepare_data(data: np.ndarray, seq_length: int, train_split: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """Normalizes data, creates sequences, and splits into train/test sets.
    
    Args:
        data (np.ndarray): Raw input data.
        seq_length (int): Length of the sliding window.
        train_split (float): Fraction of data to use for training.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
            X_train, y_train, X_test, y_test, and the scaler object.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))

    # Convert to sequences
    X, y = create_sequences(data_normalized, seq_length)

    # Split into train and test
    train_size = int(len(X) * train_split)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test, scaler
