import pandas as pd
from sklearn.model_selection import train_test_split

csv_file = r"C:\Users\dmk it win\Downloads\activity_recognition\clips_labels.csv"

df = pd.read_csv(csv_file)

# Stratified split to keep class balance
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_df.to_csv(r"C:\Users\dmk it win\Downloads\activity_recognition\train_clips.csv", index=False)
val_df.to_csv(r"C:\Users\dmk it win\Downloads\activity_recognition\val_clips.csv", index=False)

print("Train/Validation CSVs created successfully!")
