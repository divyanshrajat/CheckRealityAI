"""
CheckReality AI - Cache Video Face Crops for Backbone Fine-Tuning
Extracts 4 face crops per video from train and tune splits in video_deepfake.csv
Saves compact uint8 tensors:
- training/manifests/backbone_train_frames.pt (5600 frames, ~840MB)
- training/manifests/backbone_tune_frames.pt  (1200 frames, ~180MB)
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import json

def extract_crops_from_video(args):
    vpath, lbl, num_frames, box = args
    if not os.path.exists(vpath):
        return None
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    
    indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)
    idx_set = set(indices)
    max_idx = max(indices)
    
    fx, fy, fw, fh = box # normalized [0, 1]
    
    crops = []
    for i in range(max_idx + 1):
        if i in idx_set:
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                # 1. Full-face bounding crop with 20% context margin
                pad_w, pad_h = int(0.20 * fw * w), int(0.20 * fh * h)
                x1 = max(0, int(fx * w) - pad_w)
                y1 = max(0, int(fy * h) - pad_h)
                x2 = min(w, int((fx + fw) * w) + pad_w)
                y2 = min(h, int((fy + fh) * h) + pad_h)
                crop1 = frame[y1:y2, x1:x2]
                if crop1.size > 0:
                    crop1 = cv2.resize(crop1, (224, 224))
                    crop1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)
                    crops.append(np.ascontiguousarray(crop1.transpose(2, 0, 1)))
                else:
                    crops.append(np.zeros((3, 224, 224), dtype=np.uint8))
                
                # 2. Targeted mouth & lower-face crop (captures NeuralTextures and Face2Face lip/teeth/expression)
                mx1 = max(0, int((fx + 0.10 * fw) * w))
                my1 = max(0, int((fy + 0.45 * fh) * h))
                mx2 = min(w, int((fx + 0.90 * fw) * w))
                my2 = min(h, int((fy + 1.05 * fh) * h))
                crop2 = frame[my1:my2, mx1:mx2]
                if crop2.size > 0:
                    crop2 = cv2.resize(crop2, (224, 224))
                    crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)
                    crops.append(np.ascontiguousarray(crop2.transpose(2, 0, 1)))
                else:
                    crops.append(np.zeros((3, 224, 224), dtype=np.uint8))
            else:
                crops.append(np.zeros((3, 224, 224), dtype=np.uint8))
                crops.append(np.zeros((3, 224, 224), dtype=np.uint8))
        else:
            cap.grab()
    cap.release()
    return crops, lbl

def main():
    csv_path = 'training/manifests/video_deepfake.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} video records from {csv_path}")

    boxes_path = 'training/manifests/face_boxes.json'
    if os.path.exists(boxes_path):
        with open(boxes_path, 'r', encoding='utf-8') as f:
            face_boxes = json.load(f)
        print(f"Loaded {len(face_boxes)} bounding boxes from {boxes_path}")
    else:
        face_boxes = {}
        print("Warning: face_boxes.json not found, using default box.")

    num_frames = 3 # 3 temporal frames * 2 crops (full + mouth) = 6 crops per video

    for split in ['tune', 'train']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        print(f"\nProcessing split '{split}': {len(split_df)} videos ({len(split_df)*num_frames*2} crops)...")
        
        total_crops = len(split_df) * num_frames * 2
        crops_arr = np.empty((total_crops, 3, 224, 224), dtype=np.uint8)
        labels_arr = np.empty((total_crops,), dtype=np.int64)
        
        tasks = [
            (row['filepath'], int(row['label_id']), num_frames, face_boxes.get(row['filepath'], [0.28, 0.15, 0.44, 0.44]))
            for _, row in split_df.iterrows()
        ]
        
        write_idx = 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, res in enumerate(ex.map(extract_crops_from_video, tasks)):
                if res is not None:
                    crops, lbl = res
                    n = len(crops)
                    for c in crops:
                        crops_arr[write_idx] = c
                        labels_arr[write_idx] = lbl
                        write_idx += 1
                if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                    print(f"  Processed {i+1}/{len(tasks)} videos ({write_idx} crops written)", flush=True)

        crops_tensor = torch.from_numpy(crops_arr[:write_idx])
        labels_tensor = torch.from_numpy(labels_arr[:write_idx])
        
        out_path = f'training/manifests/backbone_{split}_frames.pt'
        torch.save((crops_tensor, labels_tensor), out_path)
        print(f"Saved {out_path}: {crops_tensor.shape}, {crops_tensor.dtype}, {labels_tensor.shape}", flush=True)

if __name__ == '__main__':
    main()
