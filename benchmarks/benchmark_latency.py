"""
CheckReality AI - Real Latency, Parameter Count, and Size Budget Benchmark Suite (DR7)
Executes real warm-up and timed inference runs on actual models loaded with weights,
measuring mean, P50, P95, P99 latencies, parameter counts, artifact disk footprints,
and peak memory on both CPU and GPU (when available).
"""

import os
import sys
import time
import json
import platform
import psutil
import tracemalloc
import subprocess
from datetime import datetime, timezone
import torch
import torch.nn as nn
import numpy as np

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchvision.models import efficientnet_b3, resnet18, efficientnet_b0
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torchaudio
from backend.app.inference import EfficientNetLSTM, get_audio_transforms
from ml.fusion.engine import multimodal_fusion


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def get_hardware_info() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cpu_name = platform.processor() or platform.machine()
    total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return {
        "cpu": cpu_name,
        "gpu": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "total_ram_gb": total_ram_gb,
        "os": platform.platform()
    }


def get_library_versions() -> dict:
    import torchvision
    import transformers
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__
    }


def get_model_file_size_bytes(model_dir_or_file: str) -> int:
    if os.path.isfile(model_dir_or_file):
        return os.path.getsize(model_dir_or_file)
    total = 0
    if os.path.isdir(model_dir_or_file):
        for root, _, files in os.walk(model_dir_or_file):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total


def time_callable(fn, runs: int, device_str: str) -> list:
    is_cuda = "cuda" in device_str
    latencies_ms = []
    for _ in range(runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    return latencies_ms


def benchmark_device(target_device_str: str, num_warmup: int = 10, num_timed: int = 100) -> dict:
    device = torch.device(target_device_str)
    is_cuda = "cuda" in target_device_str
    torch.manual_seed(42)
    np.random.seed(42)

    results = {
        "device": target_device_str,
        "num_warmup": num_warmup,
        "num_timed": num_timed,
        "models": {}
    }

    # 1. IMAGE MODEL: EfficientNet-B3 with horizontal-flip TTA
    img_weight_path = os.path.join(PROJECT_ROOT, "models", "image_deepfake", "efficientnet_b3.pth")
    img_model = efficientnet_b3()
    img_model.classifier[1] = nn.Linear(img_model.classifier[1].in_features, 2)
    if os.path.exists(img_weight_path):
        img_model.load_state_dict(torch.load(img_weight_path, map_location=device))
    img_model.to(device)
    img_model.eval()

    img_input = torch.randn(1, 3, 224, 224, device=device)
    img_param_count = sum(p.numel() for p in img_model.parameters())
    img_disk_size_bytes = get_model_file_size_bytes(img_weight_path)

    def run_image():
        with torch.no_grad():
            out1 = img_model(img_input)
            out2 = img_model(torch.flip(img_input, dims=[3]))
            p1 = torch.softmax(out1, dim=1)[0, 1]
            p2 = torch.softmax(out2, dim=1)[0, 1]
            return (p1 + p2) / 2.0

    # Warmup
    for _ in range(num_warmup):
        run_image()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    img_latencies = time_callable(run_image, num_timed, target_device_str)

    if is_cuda:
        img_peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        img_peak_mem_mb = round(peak / (1024 ** 2), 2)

    results["models"]["image_deepfake_efficientnet_b3"] = {
        "architecture": "EfficientNet-B3 (TTA horizontal-flip)",
        "parameter_count": img_param_count,
        "disk_size_bytes": img_disk_size_bytes,
        "disk_size_mb": round(img_disk_size_bytes / (1024 ** 2), 2),
        "input_shape": [1, 3, 224, 224],
        "latency_ms": {
            "mean": round(float(np.mean(img_latencies)), 2),
            "p50": round(float(np.percentile(img_latencies, 50)), 2),
            "p95": round(float(np.percentile(img_latencies, 95)), 2),
            "p99": round(float(np.percentile(img_latencies, 99)), 2)
        },
        "peak_memory_mb": img_peak_mem_mb
    }

    # 2. VIDEO MODEL: EfficientNet-B0 + 2L LSTM (16 frames)
    vid_weight_path = os.path.join(PROJECT_ROOT, "models", "video_deepfake", "efficientnet_lstm.pth")
    vid_model = EfficientNetLSTM()
    if os.path.exists(vid_weight_path):
        vid_model.load_state_dict(torch.load(vid_weight_path, map_location=device))
    vid_model.to(device)
    vid_model.eval()

    vid_input = torch.randn(1, 16, 3, 224, 224, device=device)
    vid_param_count = sum(p.numel() for p in vid_model.parameters())
    vid_disk_size_bytes = get_model_file_size_bytes(vid_weight_path)

    def run_video():
        with torch.no_grad():
            out = vid_model(vid_input)
            return torch.softmax(out, dim=1)[0, 1]

    for _ in range(num_warmup):
        run_video()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    vid_latencies = time_callable(run_video, num_timed, target_device_str)

    if is_cuda:
        vid_peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        vid_peak_mem_mb = round(peak / (1024 ** 2), 2)

    results["models"]["video_deepfake_efficientnet_lstm"] = {
        "architecture": "EfficientNet-B0 + 2L LSTM",
        "parameter_count": vid_param_count,
        "disk_size_bytes": vid_disk_size_bytes,
        "disk_size_mb": round(vid_disk_size_bytes / (1024 ** 2), 2),
        "input_shape": [1, 16, 3, 224, 224],
        "latency_ms": {
            "mean": round(float(np.mean(vid_latencies)), 2),
            "p50": round(float(np.percentile(vid_latencies, 50)), 2),
            "p95": round(float(np.percentile(vid_latencies, 95)), 2),
            "p99": round(float(np.percentile(vid_latencies, 99)), 2)
        },
        "peak_memory_mb": vid_peak_mem_mb
    }

    # 3. AUDIO MODEL: ResNet-18 (5 s 16 kHz audio -> 128-band log-Mel)
    aud_weight_path = os.path.join(PROJECT_ROOT, "models", "audio_spoof", "audio_cnn.pth")
    aud_model = resnet18()
    aud_model.fc = nn.Linear(aud_model.fc.in_features, 2)
    if os.path.exists(aud_weight_path):
        aud_model.load_state_dict(torch.load(aud_weight_path, map_location=device))
    aud_model.to(device)
    aud_model.eval()

    mel_tf, to_db = get_audio_transforms()
    # 5 seconds at 16,000 Hz = 80,000 samples
    dummy_waveform = torch.randn(1, 16000 * 5)
    spec = to_db(mel_tf(dummy_waveform))
    spec_min, spec_max = spec.min(), spec.max()
    spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    aud_input = spec.repeat(3, 1, 1).unsqueeze(0).to(device)

    aud_param_count = sum(p.numel() for p in aud_model.parameters())
    aud_disk_size_bytes = get_model_file_size_bytes(aud_weight_path)

    def run_audio():
        with torch.no_grad():
            out = aud_model(aud_input)
            return torch.softmax(out, dim=1)[0, 1]

    for _ in range(num_warmup):
        run_audio()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    aud_latencies = time_callable(run_audio, num_timed, target_device_str)

    if is_cuda:
        aud_peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        aud_peak_mem_mb = round(peak / (1024 ** 2), 2)

    results["models"]["audio_spoof_resnet18"] = {
        "architecture": "ResNet-18 (Mel-Spectrogram)",
        "parameter_count": aud_param_count,
        "disk_size_bytes": aud_disk_size_bytes,
        "disk_size_mb": round(aud_disk_size_bytes / (1024 ** 2), 2),
        "input_shape": list(aud_input.shape),
        "latency_ms": {
            "mean": round(float(np.mean(aud_latencies)), 2),
            "p50": round(float(np.percentile(aud_latencies, 50)), 2),
            "p95": round(float(np.percentile(aud_latencies, 95)), 2),
            "p99": round(float(np.percentile(aud_latencies, 99)), 2)
        },
        "peak_memory_mb": aud_peak_mem_mb
    }

    # 4. TEXT MISINFORMATION: RoBERTa-base (512 tokens)
    misinfo_dir = os.path.join(PROJECT_ROOT, "models", "text_misinformation")
    tok = AutoTokenizer.from_pretrained(misinfo_dir)
    misinfo_model = AutoModelForSequenceClassification.from_pretrained(misinfo_dir).to(device)
    misinfo_model.eval()

    sample_512_text = "Breaking report: Forensic analysis confirms synthetic media generation. " * 35
    text_inputs = tok(sample_512_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)

    misinfo_param_count = sum(p.numel() for p in misinfo_model.parameters())
    misinfo_disk_size_bytes = get_model_file_size_bytes(misinfo_dir)

    def run_text_misinfo():
        with torch.no_grad():
            out = misinfo_model(**text_inputs).logits
            return torch.softmax(out, dim=1)[0, 1]

    for _ in range(num_warmup):
        run_text_misinfo()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    misinfo_latencies = time_callable(run_text_misinfo, num_timed, target_device_str)

    if is_cuda:
        misinfo_peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        misinfo_peak_mem_mb = round(peak / (1024 ** 2), 2)

    results["models"]["text_misinformation_roberta"] = {
        "architecture": "RoBERTa-base (Misinformation)",
        "parameter_count": misinfo_param_count,
        "disk_size_bytes": misinfo_disk_size_bytes,
        "disk_size_mb": round(misinfo_disk_size_bytes / (1024 ** 2), 2),
        "input_token_length": 512,
        "latency_ms": {
            "mean": round(float(np.mean(misinfo_latencies)), 2),
            "p50": round(float(np.percentile(misinfo_latencies, 50)), 2),
            "p95": round(float(np.percentile(misinfo_latencies, 95)), 2),
            "p99": round(float(np.percentile(misinfo_latencies, 99)), 2)
        },
        "peak_memory_mb": misinfo_peak_mem_mb
    }

    # 5. TEXT AI DETECTION: RoBERTa-base (512 tokens)
    ai_dir = os.path.join(PROJECT_ROOT, "models", "text_ai_detection")
    ai_model = AutoModelForSequenceClassification.from_pretrained(ai_dir).to(device)
    ai_model.eval()

    ai_param_count = sum(p.numel() for p in ai_model.parameters())
    ai_disk_size_bytes = get_model_file_size_bytes(ai_dir)

    def run_text_ai():
        with torch.no_grad():
            out = ai_model(**text_inputs).logits
            return torch.softmax(out, dim=1)[0, 1]

    for _ in range(num_warmup):
        run_text_ai()

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    ai_latencies = time_callable(run_text_ai, num_timed, target_device_str)

    if is_cuda:
        ai_peak_mem_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        ai_peak_mem_mb = round(peak / (1024 ** 2), 2)

    results["models"]["text_ai_detection_roberta"] = {
        "architecture": "RoBERTa-base (AI Detection)",
        "parameter_count": ai_param_count,
        "disk_size_bytes": ai_disk_size_bytes,
        "disk_size_mb": round(ai_disk_size_bytes / (1024 ** 2), 2),
        "input_token_length": 512,
        "latency_ms": {
            "mean": round(float(np.mean(ai_latencies)), 2),
            "p50": round(float(np.percentile(ai_latencies, 50)), 2),
            "p95": round(float(np.percentile(ai_latencies, 95)), 2),
            "p99": round(float(np.percentile(ai_latencies, 99)), 2)
        },
        "peak_memory_mb": ai_peak_mem_mb
    }

    # Clean up GPU memory before exiting device block
    if is_cuda:
        torch.cuda.empty_cache()

    return results


def benchmark_fusion_engine(num_calls: int = 10000) -> dict:
    test_input = {"video": 0.95, "audio": 0.88, "image": 0.92, "text": 0.10}
    # Warmup
    for _ in range(100):
        _ = multimodal_fusion(test_input)

    latencies_us = []
    t0_total = time.perf_counter()
    for _ in range(num_calls):
        t0 = time.perf_counter()
        _ = multimodal_fusion(test_input)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)  # microseconds
    t1_total = time.perf_counter()

    latencies_ms = [us / 1000.0 for us in latencies_us]
    throughput_per_sec = round(num_calls / (t1_total - t0_total), 1)

    return {
        "num_calls": num_calls,
        "input_modalities": list(test_input.keys()),
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 4),
            "p50": round(float(np.percentile(latencies_ms, 50)), 4),
            "p95": round(float(np.percentile(latencies_ms, 95)), 4),
            "p99": round(float(np.percentile(latencies_ms, 99)), 4)
        },
        "throughput_calls_per_sec": throughput_per_sec
    }


def main():
    print("=" * 70)
    print("CHECKREALITY AI: MEASURED LATENCY & SIZE BUDGET BENCHMARK (DR7)")
    print("=" * 70)

    git_hash = get_git_commit()
    utc_timestamp = datetime.now(timezone.utc).isoformat()
    hw = get_hardware_info()
    libs = get_library_versions()

    print(f"Git Commit Hash: {git_hash}")
    print(f"UTC Timestamp:   {utc_timestamp}")
    print(f"Hardware:        CPU: {hw['cpu']} | RAM: {hw['total_ram_gb']} GB")
    print(f"GPU Available:   {hw['cuda_available']} ({hw['gpu']})")
    print(f"Libraries:       Torch={libs['torch']}, Transformers={libs['transformers']}")
    print("-" * 70)

    report = {
        "benchmark_name": "DR7 Latency, Parameter Count, and Size Budget Empirical Benchmark",
        "command": "python benchmarks/benchmark_latency.py",
        "git_commit": git_hash,
        "utc_timestamp": utc_timestamp,
        "random_seed": 42,
        "hardware": hw,
        "library_versions": libs,
        "benchmarks": {}
    }

    # 1. Run CPU benchmark
    print("\n[1/3] Benchmarking Models on CPU (10 warmup, 100 timed runs)...")
    cpu_res = benchmark_device("cpu", num_warmup=10, num_timed=100)
    report["benchmarks"]["cpu"] = cpu_res
    for name, data in cpu_res["models"].items():
        lat = data["latency_ms"]
        print(f"  CPU | {name:32s} | P50={lat['p50']:6.2f}ms | P95={lat['p95']:6.2f}ms | P99={lat['p99']:6.2f}ms | Params={data['parameter_count']/1e6:5.2f}M | Size={data['disk_size_mb']:6.2f}MB")

    # 2. Run GPU benchmark if CUDA available
    if hw["cuda_available"]:
        print(f"\n[2/3] Benchmarking Models on GPU ({hw['gpu']}) (10 warmup, 100 timed runs)...")
        gpu_res = benchmark_device("cuda:0", num_warmup=10, num_timed=100)
        report["benchmarks"]["gpu"] = gpu_res
        for name, data in gpu_res["models"].items():
            lat = data["latency_ms"]
            print(f"  GPU | {name:32s} | P50={lat['p50']:6.2f}ms | P95={lat['p95']:6.2f}ms | P99={lat['p99']:6.2f}ms | VRAM={data['peak_memory_mb']:5.1f}MB")
    else:
        print("\n[2/3] GPU not available. Skipping GPU benchmark (marked NOT_MEASURED).")
        report["benchmarks"]["gpu"] = {
            "status": "NOT_MEASURED",
            "reason": "CUDA-capable GPU not detected in system"
        }

    # 3. Time multimodal_fusion() over 10,000 calls
    print("\n[3/3] Benchmarking multimodal_fusion() over 10,000 real calls...")
    fusion_res = benchmark_fusion_engine(num_calls=10000)
    report["benchmarks"]["fusion_engine"] = fusion_res
    f_lat = fusion_res["latency_ms"]
    print(f"  Fusion | 10,000 calls | Mean={f_lat['mean']:.4f}ms | P50={f_lat['p50']:.4f}ms | P95={f_lat['p95']:.4f}ms | P99={f_lat['p99']:.4f}ms | Throughput={fusion_res['throughput_calls_per_sec']} calls/sec")

    # Save output JSON
    os.makedirs(os.path.join(PROJECT_ROOT, "benchmarks"), exist_ok=True)
    out_json_path = os.path.join(PROJECT_ROOT, "benchmarks", "latency_size_budget_report.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SAVED EMPIRICAL BENCHMARK REPORT TO: {out_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
