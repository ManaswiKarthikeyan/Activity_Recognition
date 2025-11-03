import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
dataset_root = r"C:\Users\dmk it win\Downloads\activity_recognition\dataset_clips"
clips_csv = r"C:\Users\dmk it win\Downloads\activity_recognition\clips_labels.csv"
train_csv = r"C:\Users\dmk it win\Downloads\activity_recognition\train_clips.csv"
val_csv = r"C:\Users\dmk it win\Downloads\activity_recognition\val_clips.csv"

# Only these 7 actions
selected_actions = [
    'WalkingWithDog',
    'Running',
    'JumpingJack',
    'PushUps',
    'SitUps',
    'Swing',
    'Clapping'
]

# Step 1: Generate filtered CSV for selected actions
with open(clips_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['clip_path', 'label'])
    
    for action in os.listdir(dataset_root):
        if action not in selected_actions:
            continue
        action_path = os.path.join(dataset_root, action)
        if not os.path.isdir(action_path):
            continue
        for clip_file in os.listdir(action_path):
            if clip_file.endswith(".npy"):
                clip_path = os.path.join(action_path, clip_file)
                writer.writerow([clip_path, action])

print("✅ Filtered clips CSV created successfully!")

# Step 2: Split into train and validation CSVs
df = pd.read_csv(clips_csv)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

print("✅ Train and Validation CSVs created successfully!")
