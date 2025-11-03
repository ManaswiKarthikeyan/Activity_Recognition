# visualize.py
import argparse, json
import cv2, imageio, os
from data.clip_dataset import load_video_frames
from inference import sliding_predictions, load_model, predict_segments
import numpy as np

def overlay_text(frame, text, pos=(10,30), bg_alpha=0.6):
    # draw semi-transparent background for readability
    x,y = pos
    h = 30
    w = 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-5,y-20), (x+w,y+h), (0,0,0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1-bg_alpha, 0, frame)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return frame

def make_demo(model_path, video_path, out_path='demo.gif', clip_len=16, stride=8, target_fps=15, resize=112):
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model, classes = load_model(model_path, device)
    frames = load_video_frames(video_path, target_fps=target_fps, resize=(resize,resize))
    probs_list = sliding_predictions(model, frames, clip_len=clip_len, stride=stride, device=device)
    segments = predict_segments(probs_list, classes, fps=target_fps, clip_len=clip_len)
    # create label per frame (simple: for each frame index, find segments covering it)
    frame_labels = [('', 0.0) for _ in frames]
    for seg in segments:
        s_frame = int(seg['start_time'] * target_fps)
        e_frame = int(seg['end_time'] * target_fps)
        for i in range(max(0,s_frame), min(len(frames), e_frame+1)):
            frame_labels[i] = (seg['label'], seg['score'])
    # overlay and save frames
    vis_frames = []
    for i, f in enumerate(frames):
        frame = f.copy()
        label, score = frame_labels[i]
        ts = f"t={i/target_fps:.2f}s"
        if label:
            text = f"{label} {score:.2f} | {ts}"
        else:
            text = f"no-action | {ts}"
        # convert RGB->BGR for cv2 putText then back to RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_bgr = overlay_text(frame_bgr, text)
        # draw timeline bar
        h, w = frame_bgr.shape[:2]
        filled = int((i/len(frames))*w)
        cv2.rectangle(frame_bgr, (0,h-10), (filled, h), (0,255,0), -1)
        vis_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    # save as gif or mp4
    ext = os.path.splitext(out_path)[1].lower()
    if ext in ['.gif']:
        imageio.mimsave(out_path, vis_frames, fps=target_fps)
    else:
        # mp4 via cv2
        h,w,_ = vis_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, target_fps, (w,h))
        for f in vis_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
    print("Saved demo to", out_path)
    return segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', default='demo.gif')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--target_fps', type=int, default=15)
    parser.add_argument('--resize', type=int, default=112)
    args = parser.parse_args()
    make_demo(args.model, args.video, out_path=args.out, clip_len=args.clip_len, stride=args.stride, target_fps=args.target_fps, resize=args.resize)
