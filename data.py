import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Example Dataset (using random data for illustration; replace with actual loading)
class EMGPoseDataset(Dataset):
    def __init__(self, signals, labels):
        """
        signals: Tensor of shape (N, C, L) where N=#samples, C=#EMG channels, L=sequence length
        labels: Tensor of shape (N,) with action/pose labels for each sequence
        """
        self.signals = signals
        self.labels = labels
    def __len__(self):
        return len(self.signals)
    def __getitem__(self, idx):
        x = self.signals[idx]
        y = self.labels[idx]
        return x, y

# Suppose we have loaded or generated data arrays `train_signals`, `train_labels`, etc.
# For demonstration, create dummy data:
num_channels = 8    # e.g., 8 EMG sensors
seq_length  = 1000  # e.g., 1000 time steps per sample
num_classes = 10    # e.g., 10 distinct action classes

# Generate random synthetic data (replace this with real dataset loading)
train_signals = torch.randn(100, num_channels, seq_length)  # 100 training samples
train_labels  = torch.randint(0, num_classes, (100,))
val_signals   = torch.randn(20, num_channels, seq_length)   # 20 validation samples
val_labels    = torch.randint(0, num_classes, (20,))

train_dataset = EMGPoseDataset(train_signals, train_labels)
val_dataset   = EMGPoseDataset(val_signals, val_labels)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=8, shuffle=False)
