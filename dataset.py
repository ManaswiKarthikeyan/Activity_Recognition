import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class VideoDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = row['clip_path']
        label = self.label2idx[row['label']]

        clip = np.load(clip_path)  # shape: (frames, H, W, C)
        clip = np.transpose(clip, (3, 0, 1, 2))  # (C, T, H, W)
        clip = clip / 255.0  # normalize to [0,1]

        return torch.tensor(clip, dtype=torch.float32), torch.tensor(label)
