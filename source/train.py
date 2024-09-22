"""
    Train module for `league-draft-analyzer`
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import constants as CONST

class TrainParams():
    """Training parameters"""
    EPOCHS = 20
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    BATCH_SIZE = 12
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    SAVE_PATH = CONST.LDA_WEIGHTS_PATH

class GameDataset(Dataset):
    """
    Custom Dataset class for game data.

    Args:
        data (list of dicts): List of game data dictionaries.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_data = self.data[idx]
        input_data = self.prepare_input_data(game_data)
        label = game_data[CONST.GAMERESULT_DATA] 
        return {"data": input_data, "label": torch.tensor(label, dtype=torch.float)}

    def prepare_input_data(self, entry:dict):
        """Class to prepare a single entry of `self.data` to tensor type, should be specifically overriden in LDANet"""
        pass

def train_model(model, dataset:GameDataset):
    """
    Train the LDANet model on the provided dataset.

    Args:
        model (nn.Module): The neural network model (LDANet instance).
        dataset (Dataset): Custom dataset of game data.
    """
    # Split dataset into training and validation sets
    train_data, val_data = train_test_split(dataset, test_size=TrainParams.TEST_SIZE, random_state=TrainParams.RANDOM_STATE)
    train_loader = DataLoader(train_data, batch_size=TrainParams.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=TrainParams.BATCH_SIZE, shuffle=False)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=TrainParams.LEARNING_RATE, weight_decay=TrainParams.WEIGHT_DECAY)
    criterion = nn.BCELoss()

    # Move model to cuda(will crash if cuda not available but we don't want cpu anyway)
    device = torch.device(CONST.DEVICE_CUDA)
    model.to(device)

    for epoch in range(TrainParams.EPOCHS):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)

            optimizer.zero_grad()  # Zero gradients before each backward pass
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels.unsqueeze(1))  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)

        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{TrainParams.EPOCHS} - Training loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set after each epoch
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), TrainParams.SAVE_PATH)
    print(f"Model saved at {TrainParams.SAVE_PATH}")

def evaluate_model(model, val_loader, criterion):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function (e.g., BCELoss).
        device (torch.device): The device (CPU or GPU).

    Returns:
        tuple(float, float): Validation loss and validation accuracy.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in val_loader:
            inputs, labels = batch["data"].to(CONST.DEVICE_CUDA), batch["label"].to(CONST.DEVICE_CUDA)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)

    val_accuracy = correct_predictions / total_predictions
    return val_loss, val_accuracy

