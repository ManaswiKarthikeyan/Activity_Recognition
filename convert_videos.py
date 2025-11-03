import os
import subprocess

# Path to UCF101 dataset
root = r"C:\Users\dmk it win\Downloads\activity_recognition\UCF101\UCF-101"

# Only these actions will be converted
selected_actions = [
    "WalkingWithDog",
    "Running",
    "JumpingJack",
    "PushUps",
    "SitUps",
    "Swing",
    "Clapping"
]

# Path to ffmpeg executable (if not in PATH, set full path)
ffmpeg_path = "ffmpeg"  # if ffmpeg is in your PATH, otherwise use full path

print("🎬 Starting conversion for selected actions...\n")

for action in selected_actions:
    action_path = os.path.join(root, action)
    if not os.path.isdir(action_path):
        print(f"⚠️ Folder not found: {action}")
        continue

    print(f"Processing folder: {action}")

    video_files = [f for f in os.listdir(action_path) if f.endswith(".avi")]
    print(f"Found {len(video_files)} videos in {action}")

    for video_file in video_files:
        in_path = os.path.join(action_path, video_file)
        out_path = os.path.splitext(in_path)[0] + ".mp4"

        # Skip if already converted
        if os.path.exists(out_path):
            print(f"✅ Already converted: {video_file}")
            continue

        print(f"Converting {video_file} ...")
        subprocess.run([
            ffmpeg_path, "-i", in_path,
            "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
            "-y", out_path
        ])
print("\n✅ Conversion complete for all selected actions!")

