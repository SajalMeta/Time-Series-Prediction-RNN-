import torch
import numpy as np
import random
from data_loader import generate_sine_wave_data, prepare_data
from model import TimeSeriesLSTM
from train import train_model
from visualize import plot_loss, plot_predictions

# --- Configuration ---
SEQ_LENGTH = 50
HIDDEN_SIZE = 50
NUM_LAYERS = 1
OUTPUT_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
SEED = 42

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (optional, slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 0. Setup
    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # 1. Prepare Data
    print("Generating and preparing data...")
    # Generate 1000 data points of a sine wave
    data = generate_sine_wave_data(seq_length=1000)
    
    X_train, y_train, X_test, y_test, scaler = prepare_data(data, SEQ_LENGTH)
    
    # Check shapes
    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # 2. Initialize Model
    print("Initializing LSTM Model...")
    model = TimeSeriesLSTM(input_size=1, 
                           hidden_size=HIDDEN_SIZE, 
                           output_size=OUTPUT_SIZE, 
                           num_layers=NUM_LAYERS)
    
    # 3. Train Model
    print("Starting training...")
    model, loss_history = train_model(model, 
                                      X_train, 
                                      y_train, 
                                      num_epochs=NUM_EPOCHS, 
                                      learning_rate=LEARNING_RATE,
                                      device=device)
    
    # 4. Evaluation and Visualization
    print("Visualizing results...")
    plot_loss(loss_history)
    plot_predictions(model, X_test, y_test, scaler, device=device)
    
    print("Done! Check 'loss_curve.png' and 'prediction_result.png'.")

if __name__ == "__main__":
    main()
