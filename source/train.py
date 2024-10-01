"""
    Train module for `league-draft-analyzer`
"""

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import constants as CONST

class TrainParams():
    """Training parameters"""
    EPOCHS = 13
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    BATCH_SIZE = 12 # 12
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001 # fuck this param, never again
    SAVE_PATH = CONST.LDA_WEIGHTS_PATH

def train_model(model, dataset):
    """
    Train the LDANet model on the provided dataset.

    Args:
        model (nn.Module): The neural network model (LDANet instance).
        dataset (LDADataset): Custom dataset of game data.
    """
    # Split dataset into training and validation sets
    train_data, val_data = train_test_split(dataset, test_size=TrainParams.TEST_SIZE, random_state=TrainParams.RANDOM_STATE)
    train_loader = DataLoader(train_data, batch_size=TrainParams.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=TrainParams.BATCH_SIZE, shuffle=False)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=TrainParams.LEARNING_RATE, weight_decay=TrainParams.WEIGHT_DECAY)
    criterion = nn.BCELoss()

    # Move model to the appropriate device
    device = torch.device(CONST.DEVICE_CUDA)
    model.to(device)

    training_start_time = time.time()
    for epoch in range(TrainParams.EPOCHS):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        epoch_start_time = time.time()
        count_1 = 0
        count_0 = 0
        count_gt_75 = 0
        count_lt_25 = 0

        for batch in train_loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)
            optimizer.zero_grad()  # Zero gradients before each backward pass

            outputs = list()
            for game in range(len(inputs)):
                output = model(inputs[game])
                item_output = output.item()
                if item_output > 0.5:
                    if item_output > 0.75:
                        count_gt_75 += 1
                    count_1 += 1
                else:
                    if item_output < 0.25:
                        count_lt_25 += 1
                    count_0 += 1
                outputs.append(output)
            outputs = torch.stack(outputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            # Accumulate loss and calculate accuracy
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()

            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)

        # Print weights and biases for epoch
        print("#"*120)
        model.monitor_weights()

        # Compute average loss and accuracy
        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{TrainParams.EPOCHS} - Training loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Time taken: {round((time.time() - epoch_start_time) / 60, 2)}min")

        # Evaluate on validation set after each epoch
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}")

        # Print info regarding networks predictions
        print(f"count_0={count_0};count_1={count_1};count_lt_25={count_lt_25};count_gt_75={count_gt_75}")

    # Print time taken
    print("#"*120)
    minutes_taken = round((time.time() - training_start_time) / 60, 2)
    print(f"# Training complete, took: {minutes_taken}min")
    print(f"# Epochs per minute: {TrainParams.EPOCHS / minutes_taken}")

    # Save the trained model
    torch.save(model.state_dict(), TrainParams.SAVE_PATH)
    print(f"# Model saved at {TrainParams.SAVE_PATH}")
    print("#"*120)

def evaluate_model(model, val_loader, criterion, device):
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
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in val_loader:
            inputs, labels = batch["data"].to(device), batch["label"].to(device)

            outputs = list()
            for game in range(len(inputs)):
                outputs.append(model(inputs[game]))
            outputs = torch.stack(outputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item() * inputs.size(0)

            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_predictions += labels.size(0)

    # Compute average loss and accuracy
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct_predictions / total_predictions
    return val_loss, val_accuracy

