import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    """LSTM model for time series prediction.
    
    Architecture:
        1. LSTM Layer: Captures temporal dependencies.
        2. Fully Connected (Linear) Layer: Maps LSTM output to a single value.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 50, output_size: int = 1, num_layers: int = 1):
        """
        Args:
            input_size (int): Number of features in input (e.g., 1 for univariate).
            hidden_size (int): Number of hidden units in LSTM.
            output_size (int): Number of output predictions.
            num_layers (int): Number of stacked LSTM layers.
        """
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True expects input shape: (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input data of shape (batch, seq_len, input_size)
        
        Returns:
            torch.Tensor: Prediction of shape (batch, output_size)
        """
        # Initialize hidden and cell states
        # We use x.size(0) to get the current batch size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the LAST time step
        # out[:, -1, :] extracts the last output of the sequence for each sample in the batch
        out = self.fc(out[:, -1, :])
        return out
