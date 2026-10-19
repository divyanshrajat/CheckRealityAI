"""
High-throughput parallel feature extraction for Video Deepfake detection.
Extracts 16-frame spatial embeddings [N, 16, 1280] using fine-tuned EfficientNet-B0
with center-face cropping, for Train (700), Tune (152), and Test (148) splits.
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from concurrent.futures import ThreadPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = Compose([
    ToTensor(),
    Resize((224, 224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_backbone():
    model = efficientnet_b0(weights=None)
    model.classifier = torch.nn.Identity()
    ft_path = 'models/video_deepfake/finetuned_b0_features.pth'
    if os.path.exists(ft_path):
        model.features.load_state_dict(torch.load(ft_path, map_location='cpu'))
        print(f"Loaded fine-tuned backbone weights from {ft_path}")
    else:
        raise FileNotFoundError(f"Missing {ft_path}!")
    model.to(device)
    model.eval()
    return model

import json

boxes_path = 'training/manifests/face_boxes.json'
if os.path.exists(boxes_path):
    with open(boxes_path, 'r', encoding='utf-8') as f:
        face_boxes = json.load(f)
else:
    face_boxes = {}

def extract_frames_from_video(vpath):
    if not os.path.exists(vpath):
        return torch.zeros((16, 3, 224, 224)), torch.zeros((16, 3, 224, 224))
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return torch.zeros((16, 3, 224, 224)), torch.zeros((16, 3, 224, 224))
    indices = np.linspace(0, max(0, total - 1), 16, dtype=int)
    idx_set = set(indices)
    max_idx = max(indices)
    
    fx, fy, fw, fh = face_boxes.get(vpath, [0.28, 0.15, 0.44, 0.44])
    
    full_dict = {}
    interior_dict = {}
    for i in range(max_idx + 1):
        if i in idx_set:
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                # 1. Full-face bounding crop with 20% margin
                pad_w, pad_h = int(0.20 * fw * w), int(0.20 * fh * h)
                x1 = max(0, int(fx * w) - pad_w)
                y1 = max(0, int(fy * h) - pad_h)
                x2 = min(w, int((fx + fw) * w) + pad_w)
                y2 = min(h, int((fy + fh) * h) + pad_h)
                crop1 = frame[y1:y2, x1:x2]
                if crop1.size > 0:
                    crop1 = cv2.resize(crop1, (224, 224))
                    crop1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)
                    full_dict[i] = transform(crop1)
                else:
                    full_dict[i] = torch.zeros((3, 224, 224))
                
                # 2. Targeted mouth & lower-face crop
                mx1 = max(0, int((fx + 0.10 * fw) * w))
                my1 = max(0, int((fy + 0.45 * fh) * h))
                mx2 = min(w, int((fx + 0.90 * fw) * w))
                my2 = min(h, int((fy + 1.05 * fh) * h))
                crop2 = frame[my1:my2, mx1:mx2]
                if crop2.size > 0:
                    crop2 = cv2.resize(crop2, (224, 224))
                    crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)
                    interior_dict[i] = transform(crop2)
                else:
                    interior_dict[i] = torch.zeros((3, 224, 224))
            else:
                full_dict[i] = torch.zeros((3, 224, 224))
                interior_dict[i] = torch.zeros((3, 224, 224))
        else:
            cap.grab()
    cap.release()
    
    f1 = torch.stack([full_dict.get(idx, torch.zeros((3, 224, 224))) for idx in indices])
    f2 = torch.stack([interior_dict.get(idx, torch.zeros((3, 224, 224))) for idx in indices])
    return f1, f2

def extract_split(df_split, split_name, backbone, batch_size=16):
    print(f"\n==================================================")
    print(f"Extracting features for split: {split_name} ({len(df_split)} videos)")
    print(f"==================================================")
    
    paths = df_split['filepath'].tolist()
    labels = df_split['label_id'].tolist()
    
    all_features = []
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(0, len(paths), batch_size):
            chunk_paths = paths[i : i + batch_size]
            results = list(executor.map(extract_frames_from_video, chunk_paths))
            tensors_full = [r[0] for r in results]
            tensors_int = [r[1] for r in results]
            
            b_full = torch.stack(tensors_full).to(device)
            b_int = torch.stack(tensors_int).to(device)
            b, t, c, h, w = b_full.shape
            
            with torch.no_grad():
                f_full = backbone(b_full.view(-1, c, h, w)).view(b, t, -1)
                f_int = backbone(b_int.view(-1, c, h, w)).view(b, t, -1)
                # Concatenate full-face (1280-d) and mouth (1280-d) -> clean 2560-d dual-stream representation
                chunk_feats = torch.cat([f_full, f_int], dim=-1).cpu() # [B, 16, 2560]
                if torch.isnan(chunk_feats).any():
                    raise ValueError(f"NaN detected in chunk {i}-{i+len(chunk_paths)} of split {split_name}!")
                all_features.append(chunk_feats)
                
            elapsed = time.time() - t0
            done = min(i + batch_size, len(paths))
            rate = done / elapsed if elapsed > 0 else 0
            print(f"[{split_name.upper()}] {done}/{len(paths)} videos processed ({rate:.1f} vid/s, elapsed {elapsed:.1f}s)", flush=True)
            
    final_feats = torch.cat(all_features, dim=0)
    final_labels = torch.tensor(labels, dtype=torch.long)
    print(f"Split {split_name} extraction complete! Tensor shape: {final_feats.shape}, Labels: {final_labels.shape}, NaNs: {torch.isnan(final_feats).any().item()}")
    return final_feats, final_labels

def main():
    backbone = load_backbone()
    manifest_path = 'training/manifests/video_deepfake.csv'
    df = pd.read_csv(manifest_path)
    
    # 1. Train split
    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    train_feats, train_labels = extract_split(df_train, 'train', backbone)
    torch.save((train_feats, train_labels), 'training/manifests/video_train_feats.pt')
    print("Saved training/manifests/video_train_feats.pt")
    
    # 2. Tune split
    df_tune = df[df['split'] == 'tune'].reset_index(drop=True)
    tune_feats, tune_labels = extract_split(df_tune, 'tune', backbone)
    torch.save((tune_feats, tune_labels), 'training/manifests/video_tune_feats.pt')
    torch.save((tune_feats, tune_labels), 'training/manifests/video_val_feats.pt')
    print("Saved training/manifests/video_tune_feats.pt (and video_val_feats.pt)")
    
    # 3. Test split
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    test_feats, test_labels = extract_split(df_test, 'test', backbone)
    torch.save((test_feats, test_labels), 'training/manifests/video_test_feats.pt')
    print("Saved training/manifests/video_test_feats.pt")
    
    print("\nALL 1,000 VIDEO FEATURES EXTRACTED AND SAVED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
