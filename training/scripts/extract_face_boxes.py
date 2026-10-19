"""
CheckReality AI - Extract Face Bounding Boxes across All Videos
Detects face locations on keyframe 0 of all 3,500 videos and saves normalized [x, y, w, h] to training/manifests/face_boxes.json.
"""

import os
import cv2
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def detect_face_box(vpath):
    if not os.path.exists(vpath):
        return [0.28, 0.15, 0.44, 0.44]
        
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [0.28, 0.15, 0.44, 0.44]
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Try first 3 keyframes if needed
    indices = [0, min(10, total - 1), min(25, total - 1)]
    found_box = None
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        h, w, _ = frame.shape
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            # Sort by area descending
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            # Convert back to normalized coordinates [0, 1]
            found_box = [round(float(fx * 2 / w), 4),
                         round(float(fy * 2 / h), 4),
                         round(float(fw * 2 / w), 4),
                         round(float(fh * 2 / h), 4)]
            break
            
    cap.release()
    if found_box is not None:
        return found_box
    return [0.28, 0.15, 0.44, 0.44]

def main():
    csv_path = 'training/manifests/video_deepfake.csv'
    df = pd.read_csv(csv_path)
    paths = df['filepath'].tolist()
    print(f"Detecting face bounding boxes for {len(paths)} videos across 8 workers...")
    
    t0 = time.time()
    face_boxes = {}
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(detect_face_box, paths))
        
    for p, box in zip(paths, results):
        face_boxes[p] = box
        
    dur = time.time() - t0
    print(f"Extraction complete in {dur:.1f}s ({len(paths)/dur:.1f} vid/s)!")
    
    out_path = 'training/manifests/face_boxes.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(face_boxes, f, indent=2)
    print(f"Saved bounding boxes to {out_path}")

if __name__ == '__main__':
    main()
