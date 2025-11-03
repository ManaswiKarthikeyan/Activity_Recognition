**Activity Recognition Project**


This project implements a 3D CNN-based action recognition system using the UCF101 dataset. The model classifies short video clips into selected action categories.
---
Currently, the project focuses on the following 7 actions:
*WalkingWithDog
*Running
*JumpingJack
*PushUps
*SitUps
*Swing
*Clapping
---
The system supports:
*Video conversion to .mp4
*Splitting videos into clips
*Generating train/validation CSVs
*Training a 3D CNN model (activity_model.pth)
*Inference on new video clips
*Evaluation with accuracy and confusion matrix
---
##Folder Structure
activity_recognition/
│
├─ UCF101/                          # Original UCF101 videos
│   └─ UCF-101/
│       ├─ WalkingWithDog/
│       ├─ Running/
│       └─ ...
│
├─ dataset_clips/                    # Generated clips (.npy files)
│   ├─ WalkingWithDog/
│   ├─ Running/
│   └─ ...
│
├─ clips_labels.csv                  # Clip paths + labels
├─ train_clips.csv                   # Training set CSV
├─ val_clips.csv                     # Validation set CSV
│
├─ convert_videos.py                 # Convert .avi to .mp4
├─ split_videos.py                   # Split videos into fixed-length clips
├─ generate_csv.py                   # Create `clips_labels.csv` from clips
├─ generate_train_val_csv.py         # Create filtered train/val CSVs
├─ train.py                          # Train 3D CNN on selected actions
├─ train_small.py                    # Optional: faster training for 7 actions
├─ inference.py                      # Run inference on new video
├─ evaluate.py                       # Evaluate model on validation set
├─ model.py                          # 3D CNN model definition
├─ activity_model.pth                # Trained PyTorch model
└─ README.md                         # This file

Requirements
Python 3.10+
PyTorch 2.0+
OpenCV (cv2)
NumPy
pandas
tqdm
scikit-learn
matplotlib
seaborn
---

Usage
1. Convert Videos

Convert .avi videos to .mp4:
python convert_videos.py

2. Split Videos into Clips

Split MP4 videos into fixed-length clips:
python split_videos.py

3. Generate CSV Files

Generate clip labels CSV:
python generate_csv.py
Generate filtered train/validation CSVs for 7 actions:
python generate_train_val_csv.py

4. Train Model

Train the 3D CNN:
python train.py
Use train_small.py for faster training on only the 7 actions.

5. Run Inference

Predict actions on a new video:
python inference.py

6. Evaluate Model

Evaluate model performance on validation set:
python evaluate.py


Outputs:

Validation accuracy
Confusion matrix
Precision, recall, F1-score

