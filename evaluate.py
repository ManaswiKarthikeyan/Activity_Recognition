import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from model import Simple3DCNN  # your model.py

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_file = r"C:\Users\dmk it win\Downloads\activity_recognition\val_clips.csv"
num_classes = 7  # your selected 7 actions
batch_size = 8

# Define dataset
class ClipDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip = np.load(row['clip_path'])  # shape: (frames, H, W, C)
        clip = np.transpose(clip, (3, 0, 1, 2))  # to (C, D, H, W) for 3D CNN
        clip = torch.tensor(clip, dtype=torch.float32) / 255.0
        label = self.label_map[row['label']]
        return clip, label

# Load dataset
val_dataset = ClipDataset(csv_file)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = Simple3DCNN(num_classes)
model.load_state_dict(torch.load("activity_model.pth", map_location=device))
model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for clips, labels in val_loader:
        clips = clips.to(device)
        labels = labels.to(device)
        outputs = model(clips)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f"Validation Accuracy: {accuracy*100:.2f}%")

# Optional: confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=val_dataset.label_map.keys(), yticklabels=val_dataset.label_map.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
