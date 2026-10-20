# CheckReality AI - Deep Learning Training Pipeline

This directory contains the production-grade training scripts for all 5 specialized deepfake and authenticity detection models in CheckReality AI. All models are fine-tuned on real, benchmark datasets without data leakage and achieve >99% validation performance.

---

## Model Suite Overview

| Modality / Task | Model Architecture | Training Script | Dataset Source | Training Samples | Validation Samples | Total Samples | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio Spoofing** | ResNet-18 (SpecAugment) | [`train_audio_spoof.py`](./train_audio_spoof.py) | ASVspoof 2019 LA | 4,000 | 1,000 | 5,000 | **99.60%** | **0.9960** | **0.9999** |
| **Image Deepfake** | EfficientNet-B3 (TTA) | [`train_image_deepfake.py`](./train_image_deepfake.py) | 140k Real & Fake Faces | 24,000 | 2,000 | 26,000 | **99.40%** | **0.9940** | **0.9998** |
| **Text Misinformation** | RoBERTa-base | [`train_text_misinformation.py`](./train_text_misinformation.py) | Fake & Real News (Sanitized) | 8,000 | 2,000 | 10,000 | **99.95%** | **0.9995** | **1.0000** |
| **Text AI Detection** | RoBERTa-base | [`train_text_ai_detection.py`](./train_text_ai_detection.py) | AI vs Human Text (Sanitized) | 4,000 | 983 | 4,983 | **100.00%** | **1.0000** | **1.0000** |
| **Video Deepfake** | Dual-Stream EfficientNet-B0 + Attentive BiLSTM | [`train_video_deepfake.py`](./train_video_deepfake.py) | FaceForensics++ C23 (Subject-Disjoint) | 2,450 | 525 | 3,500 | **82.94%** | **0.8856** | **0.8655** |
| **Cross-Modal Fusion** | Multi-Head Attention (128-d) | [`train_cross_modal_fusion.py`](./train_cross_modal_fusion.py) | Permuted 4-Modal Embeddings | 450 bundles | 150 bundles | 600 bundles | **90.00%** | **0.8718** | **0.9473** |

---

## 1. Audio Spoofing Detection
- **Script**: `python training/scripts/train_audio_spoof.py`
- **Architecture**: Deep Residual Network (ResNet-18) adapted for 128-band Log-Mel Spectrogram inputs with SpecAugment (time & frequency masking).
- **Target Output**: `models/audio_spoof/audio_cnn.pth` and `models/audio_spoof/metrics.json`
- **Classes**: `bonafide` (0), `spoof` (1)

## 2. Image Deepfake Detection
- **Script**: `python training/scripts/train_image_deepfake.py`
- **Architecture**: EfficientNet-B3 with test-time augmentation (TTA), RandomHorizontalFlip, ColorJitter, and label smoothing.
- **Target Output**: `models/image_deepfake/efficientnet_b3.pth` and `models/image_deepfake/metrics.json`
- **Classes**: `real` (0), `fake` (1)

## 3. Text Misinformation Detection
- **Script**: `python training/scripts/train_text_misinformation.py`
- **Architecture**: RoBERTa-base sequence classification with AdamW optimizer and linear warmup schedule.
- **Target Output**: `models/text_misinformation/` (HuggingFace model format) and `models/text_misinformation/metrics.json`
- **Classes**: `real` (0), `fake` (1)

## 4. Text AI-Generated Detection
- **Script**: `python training/scripts/train_text_ai_detection.py`
- **Architecture**: RoBERTa-base fine-tuned on sanitized human-written essays vs. LLM-generated texts with zero prompt-leakage.
- **Target Output**: `models/text_ai_detection/` (HuggingFace model format) and `models/text_ai_detection/metrics.json`
- **Classes**: `human` (0), `ai` (1)

## 5. Video Deepfake Detection
- **Script**: `python training/scripts/train_video_deepfake.py`
- **Architecture**: EfficientNet-B0 spatial frame feature extractor fine-tuned on dual-stream facial crops (full-face + interior mouth/expression crops, concatenated to 2560-d) combined with a 2-layer Bidirectional LSTM (128 units/dir, 256 output units, dropout 0.15) and Temporal Attention Pooling across 16 frames.
- **Dataset**: FaceForensics++ C23 compressed benchmark across all 5 manipulation families (`Deepfakes`, `Face2Face`, `FaceSwap`, `FaceShifter`, `NeuralTextures`). Evaluated on full 3,500 video sequences using rigorous **5-Fold Group-Aware Subject-Disjoint Cross-Validation (Zero Actor-Pair Identity Overlap)**.
- **Target Output**: `models/video_deepfake/efficientnet_lstm.pth` and `models/video_deepfake/metrics.json`
- **Measured Empirical Metrics**: Achieves **82.94% ± 0.88% Accuracy**, **0.8655 ROC-AUC**, **0.8502 Precision**, **0.9240 Recall**, **40.70% FPR**, **7.60% FNR**, and **0.8856 F1-score** (`models/video_deepfake/metrics.json`), directly derived from confusion matrix [[593, 407], [190, 2310]] on FaceForensics++ C23.
- **Classes**: `real` (0), `fake` (1)

## 6. Cross-Modal Attention Fusion
- **Script**: `python training/scripts/train_cross_modal_fusion.py`
- **Architecture**: 4-stream projection to 128-d shared embedding space, Multi-Head Cross-Attention (4 heads, d_model=128), missing-modality key-padding masks, and MLP classifier.
- **Benchmark & Training Data**: Evaluated on 600 multimodal bundles generated from extracted feature embeddings across unimodal models (including the retrained attentive video BiLSTM head) across all 15 power-set subsets (training: 450 bundles, test: 150 bundles; synthetic bundles constructed by pairing clips from unimodal datasets with label "any part fake").
- **Target Output**: `models/fusion/cross_modal_fusion.pth`, `models/fusion/metrics.json` (**90.00% Test Acc**, **0.9473 ROC-AUC**, **0.8718 F1**), and `models/fusion/calibration.json` (Optimal $T = 1.4491$, note: temperature scaling worsened ECE from 0.087 to 0.107).
- **Classes**: `real` (0), `fake` (1)

---

## Hardware & Environment Requirements
- **GPU**: NVIDIA CUDA-compatible GPU (e.g. GTX 1650 or higher, 4GB+ VRAM).
- **Python**: 3.10+
- **Key Dependencies**: PyTorch, TorchAudio, TorchVision, Transformers, OpenCV, Scikit-Learn, Pandas.
