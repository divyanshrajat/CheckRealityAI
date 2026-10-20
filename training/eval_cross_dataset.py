"""
CheckReality AI - Cross-Dataset Generalisation & Perturbation Robustness Benchmark (DR6)
Evaluates real models on perturbation tests (JPEG Q=50, 8kHz telephony codec downsampling,
video temporal frame dropout, text typo noise) using real files and models.
Marks missing external corpora (Celeb-DF v2, FF++ C40, In-the-Wild, LIAR) as NOT_MEASURED.
Computes real Accuracy, ROC-AUC, EER, and F1 via scikit-learn.
"""

import os
import sys
import io
import json
import random
import subprocess
from datetime import datetime, timezone
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchvision.models import efficientnet_b3, resnet18
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torchaudio

from backend.app.inference import EfficientNetLSTM, get_audio_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    diffs = np.absolute(fnr - fpr)
    if np.all(np.isnan(diffs)):
        return 0.5
    idx = np.nanargmin(diffs)
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate_binary_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    y_pred = (y_scores >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5
    eer = compute_eer(y_true, y_scores)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4)
    }


# ==============================================================================
# 1. Image Perturbation Test: JPEG Q=50 Re-encoding
# ==============================================================================
def eval_image_jpeg_perturbation(sample_count: int = 500) -> dict:
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "image_deepfake.csv")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True).iloc[:sample_count]

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

    clean_scores, pert_scores, labels = [], [], []

    for idx in range(len(val_df)):
        row = val_df.iloc[idx]
        p = os.path.join(base_dir, str(row["path"]))
        try:
            orig_img = Image.open(p).convert("RGB")
        except Exception:
            continue

        lbl = 1 if str(row["label_str"]).lower() == "fake" else 0

        # 1. Clean inference
        t_clean = img_transform(orig_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out1 = model(t_clean)
            out2 = model(torch.flip(t_clean, dims=[3]))
            p_clean = ((torch.softmax(out1, dim=1)[0, 1] + torch.softmax(out2, dim=1)[0, 1]) / 2.0).item()

        # 2. JPEG Q=50 Compression Perturbation (Real re-encoding)
        buffer = io.BytesIO()
        orig_img.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        pert_img = Image.open(buffer).convert("RGB")
        t_pert = img_transform(pert_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out1 = model(t_pert)
            out2 = model(torch.flip(t_pert, dims=[3]))
            p_pert = ((torch.softmax(out1, dim=1)[0, 1] + torch.softmax(out2, dim=1)[0, 1]) / 2.0).item()

        clean_scores.append(p_clean)
        pert_scores.append(p_pert)
        labels.append(lbl)

    y_true = np.array(labels, dtype=int)
    clean_metrics = evaluate_binary_metrics(y_true, np.array(clean_scores, dtype=float))
    pert_metrics = evaluate_binary_metrics(y_true, np.array(pert_scores, dtype=float))

    return {
        "test_name": "Image JPEG Q=50 Compression Robustness",
        "model": "EfficientNet-B3 (TTA)",
        "sample_count": len(labels),
        "sampling_method": "First 500 images of held-out validation manifest",
        "clean_baseline_metrics": clean_metrics,
        "perturbed_metrics_jpeg_q50": pert_metrics,
        "delta_auc": round(pert_metrics["roc_auc"] - clean_metrics["roc_auc"], 4)
    }


# ==============================================================================
# 2. Audio Perturbation Test: 8kHz Telephony Downsampling & Noise
# ==============================================================================
def eval_audio_codec_perturbation(sample_count: int = 300) -> dict:
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "audio_spoof.csv")
    df = pd.read_csv(manifest_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True).iloc[:sample_count]

    model_path = os.path.join(PROJECT_ROOT, "models", "audio_spoof", "audio_cnn.pth")
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    mel_tf, to_db = get_audio_transforms()
    clean_scores, pert_scores, labels = [], [], []

    for idx in range(len(val_df)):
        row = val_df.iloc[idx]
        p = str(row["filepath"])
        lbl = 1 if str(row["label"]).lower() in ["spoof", "1"] else 0
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
            continue

        # Clean Mel
        spec_c = to_db(mel_tf(waveform))
        spec_c = (spec_c - spec_c.min()) / (spec_c.max() - spec_c.min() + 1e-8)
        spec_c_t = spec_c.repeat(3, 1, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            p_clean = torch.softmax(model(spec_c_t), dim=1)[0, 1].item()

        # Perturbation: 8kHz Telephony Codec Downsampling and Upsampling back to 16kHz
        down = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)(waveform)
        up = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)(down)
        if up.shape[1] < max_s:
            up = torch.nn.functional.pad(up, (0, max_s - up.shape[1]))
        elif up.shape[1] > max_s:
            up = up[:, :max_s]

        spec_p = to_db(mel_tf(up))
        spec_p = (spec_p - spec_p.min()) / (spec_p.max() - spec_p.min() + 1e-8)
        spec_p_t = spec_p.repeat(3, 1, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            p_pert = torch.softmax(model(spec_p_t), dim=1)[0, 1].item()

        clean_scores.append(p_clean)
        pert_scores.append(p_pert)
        labels.append(lbl)

    y_true = np.array(labels, dtype=int)
    clean_metrics = evaluate_binary_metrics(y_true, np.array(clean_scores, dtype=float))
    pert_metrics = evaluate_binary_metrics(y_true, np.array(pert_scores, dtype=float))

    return {
        "test_name": "Audio Telephony 8kHz Codec Downsampling Robustness",
        "model": "ResNet-18 (SpecAugment)",
        "sample_count": len(labels),
        "sampling_method": "First 300 audio clips of held-out validation manifest",
        "clean_baseline_metrics": clean_metrics,
        "perturbed_metrics_8khz_codec": pert_metrics,
        "delta_auc": round(pert_metrics["roc_auc"] - clean_metrics["roc_auc"], 4)
    }


# ==============================================================================
# 3. Video Perturbation Test: 25% Temporal Frame Dropout
# ==============================================================================
def eval_video_frame_dropout() -> dict:
    test_feats_path = os.path.join(PROJECT_ROOT, "training", "manifests", "video_test_feats.pt")
    if os.path.exists(test_feats_path):
        features, labels = torch.load(test_feats_path, map_location=device)
        sampling_desc = f"Held-out video test split ({len(features)} sequences, 16 frames each)"
    else:
        val_feats_path = os.path.join(PROJECT_ROOT, "training", "manifests", "video_val_feats.pt")
        features, labels = torch.load(val_feats_path, map_location=device)
        sampling_desc = f"Representative video validation split ({len(features)} sequences, 16 frames each)"

    model_path = os.path.join(PROJECT_ROOT, "models", "video_deepfake", "efficientnet_lstm.pth")
    model = EfficientNetLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Clean inference
    with torch.no_grad():
        out_c, _ = model.forward_from_features(features)
        clean_scores = torch.softmax(out_c, dim=1)[:, 1].cpu().numpy()

    # Perturbed inference: Zero out 4 out of 16 frames (25% temporal dropout)
    perturbed_features = features.clone()
    np.random.seed(42)
    drop_indices = np.random.choice(16, size=4, replace=False)
    perturbed_features[:, drop_indices, :] = 0.0

    with torch.no_grad():
        out_p, _ = model.forward_from_features(perturbed_features)
        pert_scores = torch.softmax(out_p, dim=1)[:, 1].cpu().numpy()

    y_true = labels.cpu().numpy().astype(int)
    clean_metrics = evaluate_binary_metrics(y_true, clean_scores)
    pert_metrics = evaluate_binary_metrics(y_true, pert_scores)

    return {
        "test_name": "Video 25% Temporal Frame Dropout Robustness",
        "model": "EfficientNet-B0 + 2L LSTM",
        "sample_count": len(y_true),
        "sampling_method": sampling_desc,
        "clean_baseline_metrics": clean_metrics,
        "perturbed_metrics_25pct_dropout": pert_metrics,
        "delta_auc": round(pert_metrics["roc_auc"] - clean_metrics["roc_auc"], 4)
    }


# ==============================================================================
# 4. Text Perturbation Test: 5% Character Typo Injection
# ==============================================================================
def eval_text_typo_perturbation(sample_count: int = 500) -> dict:
    manifest_path = os.path.join(PROJECT_ROOT, "training", "manifests", "text_misinformation.csv")
    df = pd.read_csv(manifest_path)
    val_df_all = df[df["split"] == "val"].reset_index(drop=True)
    val_df = val_df_all.sample(n=min(sample_count, len(val_df_all)), random_state=42).reset_index(drop=True)

    model_dir = os.path.join(PROJECT_ROOT, "models", "text_misinformation")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    def inject_typos(text: str, rate: float = 0.05) -> str:
        chars = list(text)
        n_typos = int(len(chars) * rate)
        for _ in range(n_typos):
            idx = random.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                chars[idx] = chr(random.randint(97, 122))
        return "".join(chars)

    random.seed(42)
    clean_texts = val_df["text"].fillna("").astype(str).tolist()
    pert_texts = [inject_typos(t, rate=0.05) for t in clean_texts]
    labels = np.array([1 if str(lbl).lower() in ["fake", "1"] else 0 for lbl in val_df["label"]], dtype=int)

    def run_inference_batch(texts, batch_size=32):
        scores = []
        for i in range(0, len(texts), batch_size):
            b_texts = texts[i:i + batch_size]
            inputs = tok(b_texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs).logits
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                scores.extend(probs.tolist())
        return np.array(scores, dtype=float)

    clean_scores = run_inference_batch(clean_texts)
    pert_scores = run_inference_batch(pert_texts)

    clean_metrics = evaluate_binary_metrics(labels, clean_scores)
    pert_metrics = evaluate_binary_metrics(labels, pert_scores)

    return {
        "test_name": "Text 5% Typo / Character Noise Injection Robustness",
        "model": "RoBERTa-base (Misinformation)",
        "sample_count": len(labels),
        "sampling_method": "First 500 news texts of held-out validation manifest",
        "clean_baseline_metrics": clean_metrics,
        "perturbed_metrics_5pct_typos": pert_metrics,
        "delta_auc": round(pert_metrics["roc_auc"] - clean_metrics["roc_auc"], 4)
    }


def main():
    print("=" * 75)
    print("CHECKREALITY AI: REAL CROSS-DATASET & PERTURBATION EVALUATION (DR6)")
    print("Executes real inference on perturbation stress tests; logs missing corpora.")
    print("=" * 75)

    git_hash = get_git_commit()
    utc_now = datetime.now(timezone.utc).isoformat()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    report = {
        "benchmark_name": "DR6 Cross-Dataset Generalisation and Perturbation Suite",
        "command": "python training/eval_cross_dataset.py",
        "git_commit": git_hash,
        "utc_timestamp": utc_now,
        "random_seed": 42,
        "device": str(device),
        "unmeasured_external_datasets": {
            "video_celeb_df_v2": {
                "status": "NOT_MEASURED",
                "reason": "Celeb-DF v2 requires formal Data Use Agreement (DUA) and was not downloaded in the local repository."
            },
            "video_ffpp_c40": {
                "status": "NOT_MEASURED",
                "reason": "FaceForensics++ C40 extreme-compression subset was not downloaded locally; only C23 is present on disk."
            },
            "audio_in_the_wild": {
                "status": "NOT_MEASURED",
                "reason": "In-the-Wild (Mueller et al. 2022) / ASVspoof 2021 DF external evaluation audio corpus was not downloaded locally."
            },
            "text_liar_corpus": {
                "status": "NOT_MEASURED",
                "reason": "LIAR / FakeNewsNet external cross-dataset news corpus was not downloaded in the local repository."
            }
        },
        "measured_real_perturbation_benchmarks": {}
    }

    # 1. Image JPEG Perturbation
    print("\n[1/4] Running Image JPEG Q=50 Compression Stress Test (500 images)...")
    img_res = eval_image_jpeg_perturbation(sample_count=500)
    report["measured_real_perturbation_benchmarks"]["image_jpeg_q50"] = img_res
    print(f"  Clean: Acc={img_res['clean_baseline_metrics']['accuracy']:.4f}, AUC={img_res['clean_baseline_metrics']['roc_auc']:.4f}, EER={img_res['clean_baseline_metrics']['eer']:.4f}")
    print(f"  Q=50:  Acc={img_res['perturbed_metrics_jpeg_q50']['accuracy']:.4f}, AUC={img_res['perturbed_metrics_jpeg_q50']['roc_auc']:.4f}, EER={img_res['perturbed_metrics_jpeg_q50']['eer']:.4f} (Delta AUC={img_res['delta_auc']})")

    # 2. Audio Codec Perturbation
    print("\n[2/4] Running Audio Telephony 8kHz Codec Stress Test (300 audio clips)...")
    aud_res = eval_audio_codec_perturbation(sample_count=300)
    report["measured_real_perturbation_benchmarks"]["audio_8khz_codec"] = aud_res
    print(f"  Clean: Acc={aud_res['clean_baseline_metrics']['accuracy']:.4f}, AUC={aud_res['clean_baseline_metrics']['roc_auc']:.4f}, EER={aud_res['clean_baseline_metrics']['eer']:.4f}")
    print(f"  8kHz:  Acc={aud_res['perturbed_metrics_8khz_codec']['accuracy']:.4f}, AUC={aud_res['perturbed_metrics_8khz_codec']['roc_auc']:.4f}, EER={aud_res['perturbed_metrics_8khz_codec']['eer']:.4f} (Delta AUC={aud_res['delta_auc']})")

    # 3. Video Frame Dropout
    print("\n[3/4] Running Video 25% Temporal Frame Dropout Stress Test (80 video sequences)...")
    vid_res = eval_video_frame_dropout()
    report["measured_real_perturbation_benchmarks"]["video_frame_dropout_25pct"] = vid_res
    print(f"  Clean: Acc={vid_res['clean_baseline_metrics']['accuracy']:.4f}, AUC={vid_res['clean_baseline_metrics']['roc_auc']:.4f}, EER={vid_res['clean_baseline_metrics']['eer']:.4f}")
    print(f"  Drop:  Acc={vid_res['perturbed_metrics_25pct_dropout']['accuracy']:.4f}, AUC={vid_res['perturbed_metrics_25pct_dropout']['roc_auc']:.4f}, EER={vid_res['perturbed_metrics_25pct_dropout']['eer']:.4f} (Delta AUC={vid_res['delta_auc']})")

    # 4. Text Typo Injection
    print("\n[4/4] Running Text 5% Typo Noise Injection Stress Test (500 news texts)...")
    txt_res = eval_text_typo_perturbation(sample_count=500)
    report["measured_real_perturbation_benchmarks"]["text_typo_noise_5pct"] = txt_res
    print(f"  Clean: Acc={txt_res['clean_baseline_metrics']['accuracy']:.4f}, AUC={txt_res['clean_baseline_metrics']['roc_auc']:.4f}")
    print(f"  Typos: Acc={txt_res['perturbed_metrics_5pct_typos']['accuracy']:.4f}, AUC={txt_res['perturbed_metrics_5pct_typos']['roc_auc']:.4f} (Delta AUC={txt_res['delta_auc']})")

    # Save to disk
    out_path = os.path.join(PROJECT_ROOT, "models", "cross_dataset_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75)
    print(f"SAVED CROSS-DATASET & PERTURBATION REPORT TO: {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
