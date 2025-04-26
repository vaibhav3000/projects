# To run this script locally install
# pip install causal-conv1d==1.1.1
# pip install mamba-ssm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
VOCAB_SIZE = 20
SEQ_LEN = 64
D_MODEL = 256
D_STATE = 128
BATCH_SIZE = 64
LR = 8e-4
EPOCHS = 30
DROPOUT = 0.02
WD = 0.02  
SEED = 42

# Seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# Dataset Implementation
class SelectiveCopyDataset(Dataset):
    def __init__(self, num_samples, min_markers=3, max_markers=5, seq_len=SEQ_LEN):
        self.num_samples = num_samples
        self.min_markers = min_markers
        self.max_markers = max_markers
        self.seq_len = seq_len
        self.marker = 0
        self.ignore_idx = -100
        
        self.samples = []
        for idx in range(num_samples):
            rng = np.random.RandomState(SEED + idx)
            data = torch.from_numpy(
                rng.randint(1, VOCAB_SIZE, size=(self.seq_len,)))
            
            num_markers = rng.randint(self.min_markers, self.max_markers+1)
            positions = torch.from_numpy(
                rng.choice(self.seq_len, num_markers, replace=False)).sort().values
            
            inputs = data.clone()
            inputs[positions] = self.marker
            
            targets = torch.full_like(data, self.ignore_idx)
            targets[positions] = data[positions]
            
            self.samples.append((inputs, targets))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

class FinalTestDataset(SelectiveCopyDataset):
    def __init__(self, num_samples=5000, min_markers=5, max_markers=10, seq_len=128):
        super().__init__(num_samples, min_markers, max_markers, seq_len)
        # More challenging test configuration
        self.seq_len = seq_len
        self.min_markers = min_markers
        self.max_markers = max_markers

# MAMBA Architecture
class MambaCopy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.norm_in = nn.LayerNorm(D_MODEL)
        
        self.mamba1 = Mamba(d_model=D_MODEL, d_state=D_STATE, d_conv=4, expand=2)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.mamba2 = Mamba(d_model=D_MODEL, d_state=D_STATE, d_conv=4, expand=2)
        self.norm_out = nn.LayerNorm(D_MODEL)
        
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        x = self.embed(x)
        x = self.norm_in(x)
        x = self.mamba1(x) + x
        x = self.norm1(x)
        x = self.mamba2(x) + x
        x = self.norm_out(x)
        return self.head(x)

# Training Function
def train(model, train_loader, test_loader):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = logits.argmax(-1)
                mask = (targets != -100)
                correct += (preds[mask] == targets[mask]).sum().item()
                total += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        test_accuracy = correct / total if total > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['test_acc'].append(test_accuracy)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}: Loss={avg_train_loss:.4f}, Acc={test_accuracy*100:.2f}%")
    
    return history

# Comprehensive Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(-1)
            
            mask = (targets != -100).cpu().numpy()
            all_preds.extend(preds.cpu().numpy()[mask])
            all_targets.extend(targets.cpu().numpy()[mask])
    
    # Classification Report
    print("\nClassification Report:")
    
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=range(1, VOCAB_SIZE),
                yticklabels=range(1, VOCAB_SIZE))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # TPR/FPR Calculation
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    
    # TPR/FPR Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, VOCAB_SIZE), TPR, alpha=0.7, label='TPR (Recall)')
    plt.bar(range(1, VOCAB_SIZE), FPR, alpha=0.7, label='FPR')
    plt.xlabel("Token Classes")
    plt.ylabel("Rate")
    plt.title("Class-wise TPR and FPR")
    plt.legend()
    plt.show()
    
    return {
        'TPR': TPR,
        'FPR': FPR,
        'confusion_matrix': cm
    }

# Main Execution
def main():
    # Create datasets
    train_ds = SelectiveCopyDataset(10000)
    val_ds = SelectiveCopyDataset(2000)
    test_ds = FinalTestDataset()
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Initialize model
    model = MambaCopy()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train(model, train_loader, val_loader)
    
    # Training Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], 'r-o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Final Evaluation
    print("\nFinal Test Evaluation:")
    metrics = evaluate_model(model, test_loader)
    
    # Performance Summary
    print("\nTest Set Statistics:")
    print(f"- Total test samples: {len(test_ds)}")
    print(f"- Sequence length: {test_ds.seq_len}")
    print(f"- Markers per sample: {test_ds.min_markers}-{test_ds.max_markers}")
    print(f"- Average TPR: {np.mean(metrics['TPR']):.4f}")
    print(f"- Average FPR: {np.mean(metrics['FPR']):.4f}")

if __name__ == "__main__":
    main()