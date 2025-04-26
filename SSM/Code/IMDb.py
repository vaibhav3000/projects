# To run this script locally install the following:-
#pip install torch torchvision pytorch-lightning einops numpy scipy datasets scikit-learn seaborn matplotlib mamba-ssm
#


from collections import defaultdict
from typing import Optional, Mapping, Tuple, Union
import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only  # Used for logging on the main process in distributed setups
from einops import rearrange, repeat  # For tensor manipulations

# ===================== Custom Model Definitions =====================

# Define a custom model (MambaModel) that uses the Mamba layer as its building block.
class MambaModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, n_layers=2):
        super().__init__()
        # Initial projection from input dimension to model dimension
        self.proj = nn.Linear(input_dim, d_model)
        # Create a list of Mamba layers (assumes a layer type 'Mamba' exists)
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,   # model dimension
                d_state=16,        # internal state dimension (example parameter)
                d_conv=4,          # convolution dimension (example parameter)
                expand=2           # expansion factor (example parameter)
            ) for _ in range(n_layers)
        ])
        # Final fully-connected layer mapping to output dimension (e.g., number of classes)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # Project input (B, L, input_dim) to (B, L, d_model)
        x = self.proj(x)
        # Sequentially pass through each Mamba layer (maintaining shape (B, L, d_model))
        for layer in self.layers:
            x = layer(x)
        # Average pooling over the sequence length and produce output via final FC layer
        return self.fc(x.mean(dim=1))


# Define a custom model (S4Model) that uses the S4Block as its building block.
class S4Model(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, n_layers=2):
        super().__init__()
        # Project input into the model's embedding space
        self.proj = nn.Linear(input_dim, d_model)
        # Create a list of S4 layers (assumes a layer type 'S4Block' exists)
        self.layers = nn.ModuleList([
            S4Block(
                d_model=d_model,      # model dimension
                dropout=0.1,          # dropout probability for regularization
                channels=1,           # channels setting (example parameter)
                bidirectional=False   # whether S4 is bidirectional or not
            ) for _ in range(n_layers)
        ])
        # Output fully-connected layer mapping d_model to the output dimension
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Project input to model dimension
        x = self.proj(x)
        # Transpose input from (B, L, D) to (B, D, L) as expected by S4Block
        x = x.transpose(-1, -2)
        # Sequentially apply each S4Block; note that S4Block returns a tuple, but only the first element is used
        for layer in self.layers:
            x, _ = layer(x)
        # Average pooling over the sequence length dimension and apply FC layer
        return self.fc(x.mean(dim=-1))


# ===================== Data Loading and Visualization =====================

from datasets import load_dataset

# Load the IMDB dataset (for text classification) from the HuggingFace datasets library.
# A warning might be shown (e.g., for dataset caching), but it does not prevent the code from running.
dataset = load_dataset("imdb")
print(dataset["train"][0])  # Print the first training sample to verify dataset loading

# Additional imports for metric computation and plotting
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  # Ensure that matplotlib is imported for plotting


# Plot advanced metrics (ROC Curve and Confusion Matrix) after training
def plot_advanced_metrics(y_true, y_pred, model_name):
    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Set up a figure with two subplots (one for ROC, one for Confusion Matrix)
    plt.figure(figsize=(15, 5))
    
    # ROC Curve subplot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")

    # Confusion Matrix subplot
    plt.subplot(1, 2, 2)
    # Compute confusion matrix based on a threshold of 0.5 for the probabilities
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


# ===================== Training Function with Metrics Logging =====================

import time  # For timing the training epochs

def train_model_with_metrics(model, train_loader, test_loader, name, epochs=5):
    # Select device: use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up optimizer and loss function (using CrossEntropyLoss for classification)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Dictionary to store various metrics throughout training
    metrics = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'times': [], 'memory': [],
        'all_labels': [], 'all_probs': []
    }
    
    # Training loop over epochs
    for epoch in range(epochs):
        start_time = time.time()  # Start time of the epoch
        model.train()  # Set model to training mode
        train_loss, train_correct = 0, 0
        
        # Iterate over the training data batches
        for batch in train_loader:
            # Get input data and labels and move them to the selected device
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Parameter update
            
            # Aggregate training loss and count correct predictions
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        # Set model to evaluation mode (turn off dropout, etc.)
        model.eval()
        test_loss, test_correct = 0, 0
        epoch_labels = []  # To store true labels for the current epoch
        epoch_probs = []   # To store predicted probabilities for the current epoch
        
        # Evaluate on test data without calculating gradients
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                test_correct += (outputs.argmax(1) == labels).sum().item()
                
                # Apply softmax to get probabilities, then save probability of positive class
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                epoch_probs.extend(probs)
                epoch_labels.extend(labels.cpu().numpy())
        
        # Update metrics dictionary with averaged losses and accuracies per epoch
        metrics['train_loss'].append(train_loss / len(train_loader))
        metrics['test_loss'].append(test_loss / len(test_loader))
        metrics['train_acc'].append(train_correct / len(train_loader.dataset))
        metrics['test_acc'].append(test_correct / len(test_loader.dataset))
        metrics['times'].append(time.time() - start_time)
        metrics['memory'].append(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        metrics['all_labels'].extend(epoch_labels)
        metrics['all_probs'].extend(epoch_probs)
        
        # Print epoch results to the console
        print(f"{name} Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {metrics['train_loss'][-1]:.4f} | Test Loss: {metrics['test_loss'][-1]:.4f}")
        print(f"Train Acc: {metrics['train_acc'][-1]:.4f} | Test Acc: {metrics['test_acc'][-1]:.4f}")
        print("-" * 50)
    
    # After training, plot advanced metrics such as the ROC curve and confusion matrix.
    plot_advanced_metrics(
        np.array(metrics['all_labels']),
        np.array(metrics['all_probs']),
        name
    )
    
    return metrics


# ===================== Running the Models and Evaluating Results =====================

# *** Note: The following assumes that you have defined your train_loader, test_loader,
# as well as initialized the S4Model (s4_model) and MambaModel (mamba_model) instances.
# Make sure these are defined in your code before running training. ***

# Example:
# s4_model = S4Model(input_dim=<input_dimension>, output_dim=<output_classes>)
# mamba_model = MambaModel(input_dim=<input_dimension>, output_dim=<output_classes>)
# train_loader = ...  # your training DataLoader
# test_loader  = ...  # your test/validation DataLoader

# Train S4 model and capture its metrics
print("Training S4 Model...")
s4_metrics = train_model_with_metrics(s4_model, train_loader, test_loader, "S4")

# Train Mamba model and capture its metrics
print("\nTraining Mamba Model...")
mamba_metrics = train_model_with_metrics(mamba_model, train_loader, test_loader, "Mamba")

# ===================== Generating Classification Reports =====================

from sklearn.metrics import classification_report

def print_classification_report(metrics, name):
    # Convert the accumulated probabilities into binary predictions using 0.5 threshold
    y_true = metrics['all_labels']
    y_pred = (np.array(metrics['all_probs']) > 0.5).astype(int)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Print classification reports for both models
print_classification_report(s4_metrics, "S4")
print_classification_report(mamba_metrics, "Mamba")