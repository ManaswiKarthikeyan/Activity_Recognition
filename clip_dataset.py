# data/clip_dataset.py
import os, random
from glob import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

def load_video_frames(path, target_fps=15, resize=(112,112), max_frames=None):
    """
    Load frames from video path using cv2, resample to target_fps by frame dropping/duplication.
    Returns list of RGB frames (H,W,3).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    # resample
    if len(frames) == 0:
        return []
    if abs(orig_fps - target_fps) < 1e-3:
        sampled = frames
    else:
        ratio = target_fps / orig_fps
        indices = [int(i/ratio) for i in range(int(len(frames)*ratio))]
        indices = [min(i, len(frames)-1) for i in indices]
        sampled = [frames[i] for i in indices]
    if max_frames:
        sampled = sampled[:max_frames]
    return sampled

class ClipDataset(Dataset):
    """
    Expects dataset structure:
      root/
        classA/
          clip1.mp4
          clip2.mp4
        classB/
    Or root can contain folders of frames named classX_clipY/ with frames inside.
    """
    def __init__(self, root, classes=None, clip_len=16, transform=None, target_fps=15, resize=(112,112), mode='train'):
        self.root = root
        self.clip_len = clip_len
        self.transform = transform
        self.target_fps = target_fps
        self.resize = resize
        self.samples = []
        if classes is None:
            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for c in classes:
            d = os.path.join(root, c)
            for ext in ['mp4','avi','mov','mkv']:
                for path in glob(os.path.join(d, f'*.{ext}')):
                    self.samples.append((path, self.class_to_idx[c]))
        # optionally shuffle in train mode
        if mode=='train':
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = load_video_frames(path, target_fps=self.target_fps, resize=self.resize)
        # if too short, loop frames
        if len(frames) < self.clip_len:
            if len(frames)==0:
                raise RuntimeError(f"No frames in {path}")
            # loop
            while len(frames) < self.clip_len:
                frames += frames[:(self.clip_len - len(frames))]
        # temporal augmentation: random crop in time (for train)
        start = 0
        if len(frames) > self.clip_len:
            start = random.randint(0, len(frames)-self.clip_len)
        clip = frames[start:start+self.clip_len]
        clip = np.stack(clip)  # T,H,W,3
        # to tensor: (C,T,H,W)
        clip = clip.astype(np.float32) / 255.0
        clip = np.transpose(clip, (3,0,1,2))
        clip = torch.from_numpy(clip)
        if self.transform:
            clip = self.transform(clip)
        return clip, label, path
