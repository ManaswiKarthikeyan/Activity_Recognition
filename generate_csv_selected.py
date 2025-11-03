import os
import csv

root = r"C:\Users\dmk it win\Downloads\activity_recognition\dataset_clips"
selected_actions = ['WalkingWithDog','Running','JumpingJack','PushUps','SitUps','Swing','Clapping']
csv_file = r"C:\Users\dmk it win\Downloads\activity_recognition\clips_labels_selected.csv"

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['clip_path', 'label'])
    for action in selected_actions:
        action_path = os.path.join(root, action)
        if not os.path.isdir(action_path):
            print(f"⚠️ Folder not found: {action}")
            continue
        for clip_file in os.listdir(action_path):
            if clip_file.endswith(".npy"):
                clip_path = os.path.join(action_path, clip_file)
                writer.writerow([clip_path, action])

print("✅ CSV for selected actions created successfully!")
