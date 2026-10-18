"""
CheckReality AI - Model Calibration Fitting & Evaluation Suite (DR4)
Fits temperature T by minimising NLL on a held-out CALIBRATION split disjoint from
train and validation splits. Evaluates 15-bin ECE, Brier Score, and NLL before and
after calibration on the independent VALIDATION split.
Saves models/<model>/calibration.json and models/<model>/baseline_posteriors.npy.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchvision.models import efficientnet_b3, resnet18, efficientnet_b0
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import torchaudio

from backend.app.inference import EfficientNetLSTM, get_audio_transforms
from ml.fusion.calibration import compute_ece, compute_brier_score, compute_nll

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def fit_temperature_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    """Minimises NLL over T > 0 using bounded scalar optimisation."""
    z = logits
    y = labels

    def loss(T):
        scaled_z = z / max(T, 0.01)
        probs = 1.0 / (1.0 + np.exp(-scaled_z))
        probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
        return -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))

    res = minimize_scalar(loss, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def evaluate_calibration(logits: np.ndarray, labels: np.ndarray, T: float) -> dict:
    z = logits
    y = labels

    # Before (T=1.0)
    p_before = 1.0 / (1.0 + np.exp(-z))
    p_before = np.clip(p_before, 1e-12, 1.0 - 1e-12)

    # After (T=T)
    p_after = 1.0 / (1.0 + np.exp(-z / max(T, 0.01)))
    p_after = np.clip(p_after, 1e-12, 1.0 - 1e-12)

    return {
        "before": {
            "ece_15_bins": round(compute_ece(p_before.tolist(), y.tolist(), n_bins=15), 5),
            "brier_score": round(compute_brier_score(p_before.tolist(), y.tolist()), 5),
            "nll": round(compute_nll(p_before.tolist(), y.tolist()), 5)
        },
        "after": {
            "ece_15_bins": round(compute_ece(p_after.tolist(), y.tolist(), n_bins=15), 5),
            "brier_score": round(compute_brier_score(p_after.tolist(), y.tolist()), 5),
            "nll": round(compute_nll(p_after.tolist(), y.tolist()), 5)
        },
        "calibrated_posteriors": p_after
    }


# ==============================================================================
# 1. Image Model Calibration (EfficientNet-B3 with TTA)
# ==============================================================================
def calibrate_image_model(git_hash: str, utc_now: str) -> dict:
    print("\n--- [1/5] Calibrating Image Model (EfficientNet-B3) ---")
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "image_deepfake.csv")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Disjoint split: 0:1000 for CALIBRATION, 1000:2000 for VALIDATION
    cal_df = val_df.iloc[:1000].reset_index(drop=True)
    eval_df = val_df.iloc[1000:].reset_index(drop=True)

    base_dir = os.path.join(PROJECT_ROOT, "Datasets", "image", "real_vs_fake", "real-vs-fake")
    model_path = os.path.join(PROJECT_ROOT, "models", "image_deepfake", "efficientnet_b3.pth")

    model = efficientnet_b3()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img_transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_logits(subset_df):
        all_logits = []
        all_labels = []
        for idx in range(len(subset_df)):
            row = subset_df.iloc[idx]
            p = os.path.join(base_dir, str(row["path"]))
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            t = img_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out1 = model(t)
                out2 = model(torch.flip(t, dims=[3]))
                p1 = torch.softmax(out1, dim=1)[0, 1]
                p2 = torch.softmax(out2, dim=1)[0, 1]
                prob = ((p1 + p2) / 2.0).item()
                # Compute binary logit = log(prob / (1 - prob))
                prob = float(np.clip(prob, 1e-7, 1.0 - 1e-7))
                logit = np.log(prob / (1.0 - prob))
            lbl = 1 if str(row["label_str"]).lower() == "fake" else 0
            all_logits.append(logit)
            all_labels.append(lbl)
        return np.array(all_logits, dtype=float), np.array(all_labels, dtype=int)

    print(f"Running inference on Image CALIBRATION split ({len(cal_df)} samples)...")
    cal_z, cal_y = get_logits(cal_df)
    T = fit_temperature_nll(cal_z, cal_y)
    print(f"Fitted Temperature T: {T:.4f}")

    print(f"Running inference on Image VALIDATION split ({len(eval_df)} samples)...")
    val_z, val_y = get_logits(eval_df)
    eval_res = evaluate_calibration(val_z, val_y, T)

    model_dir = os.path.join(PROJECT_ROOT, "models", "image_deepfake")
    np.save(os.path.join(model_dir, "baseline_posteriors.npy"), eval_res["calibrated_posteriors"])

    report = {
        "model_name": "image_deepfake_efficientnet_b3",
        "dataset": "140k Real and Fake Faces",
        "calibration_samples": len(cal_df),
        "validation_samples": len(eval_df),
        "sampling_method": "Disjoint partition of held-out manifest (0:1000 cal, 1000:2000 val)",
        "temperature_T": round(T, 4),
        "metrics_before_calibration": eval_res["before"],
        "metrics_after_calibration": eval_res["after"],
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "command": "python training/fit_calibration.py"
    }

    with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# 2. Audio Model Calibration (ResNet-18)
# ==============================================================================
def calibrate_audio_model(git_hash: str, utc_now: str) -> dict:
    print("\n--- [2/5] Calibrating Audio Model (ResNet-18) ---")
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "audio_spoof.csv")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Disjoint split: 0:500 for CALIBRATION, 500:1000 for VALIDATION
    cal_df = val_df.iloc[:500].reset_index(drop=True)
    eval_df = val_df.iloc[500:].reset_index(drop=True)

    model_path = os.path.join(PROJECT_ROOT, "models", "audio_spoof", "audio_cnn.pth")
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    mel_tf, to_db = get_audio_transforms()

    def get_logits(subset_df):
        all_logits = []
        all_labels = []
        for idx in range(len(subset_df)):
            row = subset_df.iloc[idx]
            p = str(row["filepath"])
            try:
                waveform, sr = torchaudio.load(p)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)
                max_s = 16000 * 5
                if waveform.shape[1] > max_s:
                    waveform = waveform[:, :max_s]
                elif waveform.shape[1] < max_s:
                    waveform = torch.nn.functional.pad(waveform, (0, max_s - waveform.shape[1]))
            except Exception:
                waveform = torch.zeros(1, 16000 * 5)

            spec = to_db(mel_tf(waveform))
            spec_min, spec_max = spec.min(), spec.max()
            spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
            spec_t = spec.repeat(3, 1, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(spec_t)
                prob = torch.softmax(out, dim=1)[0, 1].item()
                prob = float(np.clip(prob, 1e-7, 1.0 - 1e-7))
                logit = np.log(prob / (1.0 - prob))

            lbl = 1 if str(row["label"]).lower() in ["spoof", "1"] else 0
            all_logits.append(logit)
            all_labels.append(lbl)
        return np.array(all_logits, dtype=float), np.array(all_labels, dtype=int)

    print(f"Running inference on Audio CALIBRATION split ({len(cal_df)} samples)...")
    cal_z, cal_y = get_logits(cal_df)
    T = fit_temperature_nll(cal_z, cal_y)
    print(f"Fitted Temperature T: {T:.4f}")

    print(f"Running inference on Audio VALIDATION split ({len(eval_df)} samples)...")
    val_z, val_y = get_logits(eval_df)
    eval_res = evaluate_calibration(val_z, val_y, T)

    model_dir = os.path.join(PROJECT_ROOT, "models", "audio_spoof")
    np.save(os.path.join(model_dir, "baseline_posteriors.npy"), eval_res["calibrated_posteriors"])

    report = {
        "model_name": "audio_spoof_resnet18",
        "dataset": "ASVspoof 2019 LA",
        "calibration_samples": len(cal_df),
        "validation_samples": len(eval_df),
        "sampling_method": "Disjoint partition of held-out manifest (0:500 cal, 500:1000 val)",
        "temperature_T": round(T, 4),
        "metrics_before_calibration": eval_res["before"],
        "metrics_after_calibration": eval_res["after"],
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "command": "python training/fit_calibration.py"
    }

    with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# 3. Video Model Calibration (EfficientNet-B0 + LSTM)
# ==============================================================================
def calibrate_video_model(git_hash: str, utc_now: str) -> dict:
    print("\n--- [3/5] Calibrating Video Model (EfficientNet-B0 + LSTM) ---")
    val_feats_path = os.path.join(PROJECT_ROOT, "training", "manifests", "video_val_feats.pt")
    features, labels = torch.load(val_feats_path, map_location=device)
    total_samples = len(labels)

    # Disjoint split: 0:40 for CALIBRATION, 40:80 for VALIDATION
    cal_feats = features[:40]
    cal_y = labels[:40].cpu().numpy().astype(int)

    val_feats = features[40:]
    val_y = labels[40:].cpu().numpy().astype(int)

    model_path = os.path.join(PROJECT_ROOT, "models", "video_deepfake", "efficientnet_lstm.pth")
    model = EfficientNetLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    def get_logits(feats):
        with torch.no_grad():
            lstm_out, _ = model.lstm(feats)
            out = model.fc(lstm_out[:, -1, :])
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            logits = np.log(probs / (1.0 - probs))
        return logits

    print(f"Running inference on Video CALIBRATION split ({len(cal_y)} samples)...")
    cal_z = get_logits(cal_feats)
    T = fit_temperature_nll(cal_z, cal_y)
    print(f"Fitted Temperature T: {T:.4f}")

    print(f"Running inference on Video VALIDATION split ({len(val_y)} samples)...")
    val_z = get_logits(val_feats)
    eval_res = evaluate_calibration(val_z, val_y, T)

    model_dir = os.path.join(PROJECT_ROOT, "models", "video_deepfake")
    np.save(os.path.join(model_dir, "baseline_posteriors.npy"), eval_res["calibrated_posteriors"])

    report = {
        "model_name": "video_deepfake_efficientnet_lstm",
        "dataset": "FaceForensics++ C23",
        "calibration_samples": len(cal_y),
        "validation_samples": len(val_y),
        "sampling_method": "Disjoint partition of held-out manifest (0:40 cal, 40:80 val)",
        "temperature_T": round(T, 4),
        "metrics_before_calibration": eval_res["before"],
        "metrics_after_calibration": eval_res["after"],
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "command": "python training/fit_calibration.py"
    }

    with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# 4. Text Misinformation Model Calibration (RoBERTa-base)
# ==============================================================================
def calibrate_text_misinformation(git_hash: str, utc_now: str) -> dict:
    print("\n--- [4/5] Calibrating Text Misinformation Model (RoBERTa-base) ---")
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "text_misinformation.csv")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Disjoint split: 0:1000 for CALIBRATION, 1000:2000 for VALIDATION
    cal_df = val_df.iloc[:1000].reset_index(drop=True)
    eval_df = val_df.iloc[1000:].reset_index(drop=True)

    model_dir = os.path.join(PROJECT_ROOT, "models", "text_misinformation")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    def get_logits(subset_df, batch_size=32):
        all_logits = []
        all_labels = []
        texts = subset_df["text"].fillna("").astype(str).tolist()
        labels = [1 if str(lbl).lower() in ["fake", "1"] else 0 for lbl in subset_df["label"]]

        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i + batch_size]
            inputs = tok(b_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs).logits
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
                logits = np.log(probs / (1.0 - probs))
                all_logits.extend(logits.tolist())

        return np.array(all_logits, dtype=float), np.array(labels, dtype=int)

    print(f"Running inference on Text Misinfo CALIBRATION split ({len(cal_df)} samples)...")
    cal_z, cal_y = get_logits(cal_df)
    T = fit_temperature_nll(cal_z, cal_y)
    print(f"Fitted Temperature T: {T:.4f}")

    print(f"Running inference on Text Misinfo VALIDATION split ({len(eval_df)} samples)...")
    val_z, val_y = get_logits(eval_df)
    eval_res = evaluate_calibration(val_z, val_y, T)

    np.save(os.path.join(model_dir, "baseline_posteriors.npy"), eval_res["calibrated_posteriors"])

    report = {
        "model_name": "text_misinformation_roberta",
        "dataset": "Fake & Real News Dataset",
        "calibration_samples": len(cal_df),
        "validation_samples": len(eval_df),
        "sampling_method": "Disjoint partition of held-out manifest (0:1000 cal, 1000:2000 val)",
        "temperature_T": round(T, 4),
        "metrics_before_calibration": eval_res["before"],
        "metrics_after_calibration": eval_res["after"],
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "command": "python training/fit_calibration.py"
    }

    with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ==============================================================================
# 5. Text AI Detection Model Calibration (RoBERTa-base)
# ==============================================================================
def calibrate_text_ai_detection(git_hash: str, utc_now: str) -> dict:
    print("\n--- [5/5] Calibrating Text AI Detection Model (RoBERTa-base) ---")
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "text_ai_detection.csv")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    # Disjoint split: 0:491 for CALIBRATION, 491:983 for VALIDATION
    cal_df = val_df.iloc[:491].reset_index(drop=True)
    eval_df = val_df.iloc[491:].reset_index(drop=True)

    model_dir = os.path.join(PROJECT_ROOT, "models", "text_ai_detection")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    def get_logits(subset_df, batch_size=32):
        all_logits = []
        all_labels = []
        texts = subset_df["text"].fillna("").astype(str).tolist()
        labels = [1 if int(lbl) == 1 or str(lbl).lower() in ["ai", "ai-generated"] else 0 for lbl in subset_df["label_id"]]

        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i + batch_size]
            inputs = tok(b_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs).logits
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
                logits = np.log(probs / (1.0 - probs))
                all_logits.extend(logits.tolist())

        return np.array(all_logits, dtype=float), np.array(labels, dtype=int)

    print(f"Running inference on Text AI Detection CALIBRATION split ({len(cal_df)} samples)...")
    cal_z, cal_y = get_logits(cal_df)
    T = fit_temperature_nll(cal_z, cal_y)
    print(f"Fitted Temperature T: {T:.4f}")

    print(f"Running inference on Text AI Detection VALIDATION split ({len(eval_df)} samples)...")
    val_z, val_y = get_logits(eval_df)
    eval_res = evaluate_calibration(val_z, val_y, T)

    np.save(os.path.join(model_dir, "baseline_posteriors.npy"), eval_res["calibrated_posteriors"])

    report = {
        "model_name": "text_ai_detection_roberta",
        "dataset": "AI vs Human Text Dataset",
        "calibration_samples": len(cal_df),
        "validation_samples": len(eval_df),
        "sampling_method": "Disjoint partition of held-out manifest (0:491 cal, 491:983 val)",
        "temperature_T": round(T, 4),
        "metrics_before_calibration": eval_res["before"],
        "metrics_after_calibration": eval_res["after"],
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "command": "python training/fit_calibration.py"
    }

    with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    print("=" * 75)
    print("CHECKREALITY AI: EMPIRICAL PROBABILITY CALIBRATION (DR4)")
    print("Fitting Temperature T by minimising NLL on disjoint calibration splits")
    print("=" * 75)

    git_hash = get_git_commit()
    utc_now = datetime.now(timezone.utc).isoformat()
    torch.manual_seed(42)
    np.random.seed(42)

    results = {
        "benchmark_name": "DR4 Probability Calibration Suite",
        "command": "python training/fit_calibration.py",
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "device": str(device),
        "multimodal_fusion_calibration_note": "Jointly labelled four-modality corpus is NOT available on disk; per Task 2 instructions, calibration is performed and evaluated per modality. Fusion-level calibration requires jointly labelled data.",
        "models": {}
    }

    # Run for all 5 models
    results["models"]["image_deepfake"] = calibrate_image_model(git_hash, utc_now)
    results["models"]["audio_spoof"] = calibrate_audio_model(git_hash, utc_now)
    results["models"]["video_deepfake"] = calibrate_video_model(git_hash, utc_now)
    results["models"]["text_misinformation"] = calibrate_text_misinformation(git_hash, utc_now)
    results["models"]["text_ai_detection"] = calibrate_text_ai_detection(git_hash, utc_now)

    summary_path = os.path.join(PROJECT_ROOT, "models", "calibration_benchmark.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("CALIBRATION SUMMARY TABLE (MEASURED ON HELD-OUT VALIDATION SPLITS):")
    print("=" * 75)
    print(f"{'Modality':<24} | {'T':<6} | {'ECE Before':<10} | {'ECE After':<10} | {'Brier Bef':<10} | {'Brier Aft':<10} | {'NLL Bef':<10} | {'NLL Aft':<10}")
    print("-" * 105)
    for k, v in results["models"].items():
        t = v["temperature_T"]
        b = v["metrics_before_calibration"]
        a = v["metrics_after_calibration"]
        print(f"{k:<24} | {t:<6.4f} | {b['ece_15_bins']:<10.5f} | {a['ece_15_bins']:<10.5f} | {b['brier_score']:<10.5f} | {a['brier_score']:<10.5f} | {b['nll']:<10.5f} | {a['nll']:<10.5f}")
    print("=" * 75)
    print(f"Full report saved to: {summary_path}")


if __name__ == "__main__":
    main()
