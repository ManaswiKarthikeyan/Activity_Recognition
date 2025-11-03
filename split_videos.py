import os
import cv2
import numpy as np
from tqdm import tqdm

# Input folder (MP4 videos)
root = r"C:\Users\dmk it win\Downloads\activity_recognition\UCF101\UCF-101"

# Output folder (clips)
output_root = r"C:\Users\dmk it win\Downloads\activity_recognition\dataset_clips"
os.makedirs(output_root, exist_ok=True)

# Clip settings
clip_length = 16  # frames per clip
resize_height, resize_width = 112, 112  # resize frames

# Only selected actions
selected_actions = [
    "WalkingWithDog",
    "Running",
    "JumpingJack",
    "PushUps",
    "SitUps",
    "Swing",
    "Clapping"
]

for action in selected_actions:
    action_path = os.path.join(root, action)
    if not os.path.isdir(action_path):
        print(f"⚠️ Folder not found: {action}")
        continue

    output_action_path = os.path.join(output_root, action)
    os.makedirs(output_action_path, exist_ok=True)

    print(f"\n🎬 Processing action: {action}")
    for video_file in tqdm(os.listdir(action_path), desc=f"{action}"):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(action_path, video_file)
        cap = cv2.VideoCapture(video_path)

        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert to RGB
            frame = cv2.resize(frame, (resize_width, resize_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            # Save every 16 frames as a clip
            if len(frames) == clip_length:
                clip_array = np.array(frames)
                clip_name = f"{os.path.splitext(video_file)[0]}_clip{count}.npy"
                np.save(os.path.join(output_action_path, clip_name), clip_array)
                frames = []
                count += 1

        cap.release()

print("\n✅ All selected actions have been split into clips!")

