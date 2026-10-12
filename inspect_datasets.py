import os
import pandas as pd
import cv2
import json
import librosa
from pathlib import Path

def create_dirs():
    dirs = [
        "training/scripts",
        "training/manifests",
        "training/experiments"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def analyze_text(base_path):
    report = {"misinformation": {}, "ai_vs_human": {}}
    
    # Misinformation
    fake_path = os.path.join(base_path, "Fake.csv")
    true_path = os.path.join(base_path, "True.csv")
    
    if os.path.exists(fake_path) and os.path.exists(true_path):
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        report["misinformation"] = {
            "fake_samples": len(fake_df),
            "true_samples": len(true_df),
            "total_samples": len(fake_df) + len(true_df),
            "columns": list(fake_df.columns),
            "file_sizes_mb": {
                "fake": os.path.getsize(fake_path) / (1024 * 1024),
                "true": os.path.getsize(true_path) / (1024 * 1024)
            }
        }
    
    # AI vs Human
    ai_path = os.path.join(base_path, "ai_vs_human_text.csv")
    if os.path.exists(ai_path):
        ai_df = pd.read_csv(ai_path)
        report["ai_vs_human"] = {
            "total_samples": len(ai_df),
            "columns": list(ai_df.columns),
            "file_size_mb": os.path.getsize(ai_path) / (1024 * 1024)
        }
        if "label" in ai_df.columns or "generated" in ai_df.columns:
            target_col = "label" if "label" in ai_df.columns else "generated"
            report["ai_vs_human"]["class_distribution"] = ai_df[target_col].value_counts().to_dict()

    return report

def analyze_images(base_path):
    report = {}
    splits = ["train.csv", "valid.csv", "test.csv"]
    
    for split in splits:
        split_path = os.path.join(base_path, "real_vs_fake", split)
        if os.path.exists(split_path):
            df = pd.read_csv(split_path)
            report[split] = {
                "samples": len(df),
                "columns": list(df.columns)
            }
            if "label" in df.columns:
                report[split]["class_distribution"] = df["label"].value_counts().to_dict()
    
    return report

def analyze_videos(base_path):
    video_dir = os.path.join(base_path, "FaceForensics++_C23")
    report = {"total_videos": 0, "formats": set(), "sample_details": []}
    
    if os.path.exists(video_dir):
        video_files = list(Path(video_dir).rglob("*.mp4")) + list(Path(video_dir).rglob("*.avi"))
        report["total_videos"] = len(video_files)
        
        # Sample first 5 videos
        for v in video_files[:5]:
            report["formats"].add(v.suffix)
            cap = cv2.VideoCapture(str(v))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                report["sample_details"].append({
                    "file": v.name,
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "duration_sec": round(duration, 2)
                })
            cap.release()
            
    report["formats"] = list(report["formats"])
    return report

def analyze_audio(base_path):
    report = {"total_files": 0, "sample_details": []}
    
    audio_files = list(Path(base_path).rglob("*.wav")) + list(Path(base_path).rglob("*.flac"))
    report["total_files"] = len(audio_files)
    
    # Sample first 5 audio files
    for a in audio_files[:5]:
        try:
            y, sr = librosa.load(str(a), sr=None, duration=5) # Load max 5 seconds for fast check
            duration = librosa.get_duration(filename=str(a))
            report["sample_details"].append({
                "file": a.name,
                "sample_rate": sr,
                "duration_sec": round(duration, 2)
            })
        except Exception as e:
            pass

    return report

if __name__ == "__main__":
    create_dirs()
    print("Starting dataset inspection...")
    
    dataset_base = "Datasets"
    report = {
        "text": analyze_text(os.path.join(dataset_base, "text")),
        "image": analyze_images(os.path.join(dataset_base, "image")),
        "video": analyze_videos(os.path.join(dataset_base, "video")),
        "audio": analyze_audio(os.path.join(dataset_base, "audio"))
    }
    
    report_path = "training/dataset_inspection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Inspection complete. Report saved to {report_path}")
