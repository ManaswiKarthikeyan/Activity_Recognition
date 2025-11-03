import os
import sys
# Import your model library
# Example: from model import load_model, predict

# ---------------- CONFIG ----------------
# Path to UCF-101 dataset
input_dir = r"C:\Users\dmk it win\Downloads\activity_recognition\UCF101\UCF-101"

# Supported video extensions
video_extensions = ['.mp4', '.avi', '.mov']

# ---------------- LOAD MODEL ----------------
# Replace this with your actual model loading code
# Example: model = load_model('path_to_model.pth')
print("Loading model...")
# model = load_model('model.pth')
print("Model loaded ✅\n")

# ---------------- COLLECT VIDEO CLIPS ----------------
clips = []
for root, dirs, files in os.walk(input_dir):
    for f in files:
        if os.path.splitext(f)[1].lower() in video_extensions:
            clips.append(os.path.join(root, f))

if len(clips) == 0:
    print("No video clips found. Please check your dataset folder.")
    sys.exit(1)

print(f"Total clips detected: {len(clips)}")
print("First 5 clips:", clips[:5], "\n")

# ---------------- RUN INFERENCE ----------------
for idx, clip_path in enumerate(clips, 1):
    print(f"[{idx}/{len(clips)}] Processing {clip_path} ...")
    # Replace with your actual inference code, for example:
    # result = model.predict(clip_path)
    # print("Predicted activity:", result)

print("\nInference complete ✅")
