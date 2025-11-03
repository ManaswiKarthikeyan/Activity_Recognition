import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# -----------------------------
# Config
# -----------------------------
selected_actions = [
    "WalkingWithDog", "Running", "JumpingJack", 
    "PushUps", "SitUps", "Swing", "Clapping"
]
max_clips_per_action = 50   # limit for quick testing
batch_size = 4
num_epochs = 5
clip_length = 16
resize_height, resize_width = 112, 112

# -----------------------------
# Dataset
# -----------------------------
class ClipDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # filter only selected actions
        self.data = self.data[self.data['label'].isin(selected_actions)]
        # optionally limit clips per action
        limited_data = []
        for action in selected_actions:
            clips = self.data[self.data['label'] == action].head(max_clips_per_action)
            limited_data.append(clips)
        self.data = pd.concat(limited_data).reset_index(drop=True)
        self.label_map = {action: idx for idx, action in enumerate(selected_actions)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clip = np.load(row['clip_path'])  # shape: (clip_length, H, W, C)
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
        clip = torch.tensor(clip, dtype=torch.float32) / 255.0
        label = self.label_map[row['label']]
        return clip, label

# -----------------------------
# Model
# -----------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*4*28*28, 128)  # adjust depending on input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Prepare data
# -----------------------------
train_dataset = ClipDataset("train_clips.csv")
val_dataset = ClipDataset("val_clips.csv")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# -----------------------------
# Training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple3DCNN(num_classes=len(selected_actions)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for i, (clips, labels) in enumerate(tqdm(train_loader, desc="Training")):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(model.state_dict(), "activity_model.pth")
print("Model saved as activity_model.pth ✅")
