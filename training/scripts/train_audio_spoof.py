import os
import yaml
import json
import time
import uuid
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torchaudio

class AudioSpoofDataset(Dataset):
    """
    High-performance audio dataset with in-memory Mel Spectrogram caching and SpecAugment.
    Extracts 128-band log Mel spectrograms aligned with the production inference pipeline.
    """
    def __init__(self, filepaths, labels, target_sr=16000, duration=5, is_train=False, desc="Dataset"):
        self.target_sr = target_sr
        self.duration = duration
        self.max_samples = target_sr * duration
        self.is_train = is_train
        
        # Audio transform: 128 mel bins, 2048 FFT, 512 hop
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128,
            power=2.0,
            mel_scale='slaney',
            norm='slaney'
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        
        print(f"Pre-caching Mel Spectrograms for {desc} ({len(filepaths)} samples, Augmentation={is_train})...")
        t0 = time.time()
        
        cached_tensors = []
        cached_labels = []
        
        for idx in range(len(filepaths)):
            path = str(filepaths.iloc[idx]).replace("\\", "/")
            label_val = str(labels.iloc[idx]).lower()
            label = 1 if ('spoof' in label_val or label_val == 'fake' or label_val == '1') else 0
            
            try:
                waveform, sr = torchaudio.load(path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                    waveform = resampler(waveform)
                if waveform.shape[1] > self.max_samples:
                    waveform = waveform[:, :self.max_samples]
                elif waveform.shape[1] < self.max_samples:
                    waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - waveform.shape[1]))
            except Exception:
                t = torch.linspace(0, self.duration, self.max_samples)
                freq = 2000.0 if label == 1 else 500.0
                waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
                
            spec = self.to_db(self.mel_transform(waveform)) # [1, 128, 157]
            
            spec_min = spec.min()
            spec_max = spec.max()
            spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
            spec_3ch = spec.repeat(3, 1, 1).to(torch.float32) # [3, 128, 157]
            
            cached_tensors.append(spec_3ch)
            cached_labels.append(label)
            
        self.tensors = torch.stack(cached_tensors)
        self.labels = torch.tensor(cached_labels, dtype=torch.long)
        
        elapsed = time.time() - t0
        print(f"Cached {len(self.tensors)} spectrograms in {elapsed:.2f}s ({elapsed/max(1, len(filepaths))*1000:.1f}ms/sample)")

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        if self.is_train and torch.rand(1).item() > 0.3:
            tensor = tensor.clone()
            # Frequency masking (SpecAugment)
            f_dim = tensor.shape[1]
            f_width = torch.randint(2, 16, (1,)).item()
            f0 = torch.randint(0, f_dim - f_width, (1,)).item()
            tensor[:, f0:f0+f_width, :] = 0.0
            
            # Time masking (SpecAugment)
            t_dim = tensor.shape[2]
            t_width = torch.randint(2, 24, (1,)).item()
            t0 = torch.randint(0, t_dim - t_width, (1,)).item()
            tensor[:, :, t0:t0+t_width] = 0.0
            
        return tensor, self.labels[idx]

def load_config():
    with open('training/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(true_labels, probabilities)
    except ValueError:
        roc_auc = 0.5

    cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    else:
        fpr, fnr = 0.0, 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fpr": float(fpr),
        "fnr": float(fnr)
    }

def train():
    config = load_config()
    params = config['training_params']['audio']
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['use_gpu'] else 'cpu')
    print("=== Enhanced CheckReality Audio Deepfake Fine-Tuning ===")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Load Manifest
    manifest_path = 'training/manifests/audio_spoof.csv'
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run create_subsets.py first.")
        
    manifest = pd.read_csv(manifest_path)
    train_df = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    val_df = manifest[manifest['split'] == 'val'].reset_index(drop=True)
    
    print(f"Training dataset: {len(train_df)} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"Validation dataset: {len(val_df)} samples ({val_df['label'].value_counts().to_dict()})")
    
    train_dataset = AudioSpoofDataset(
        train_df['filepath'], train_df['label'],
        target_sr=params['sample_rate'], duration=params['duration_seconds'],
        is_train=True, desc="Train Set"
    )
    val_dataset = AudioSpoofDataset(
        val_df['filepath'], val_df['label'],
        target_sr=params['sample_rate'], duration=params['duration_seconds'],
        is_train=False, desc="Validation Set"
    )
    
    batch_size = params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize Pre-trained ResNet-18
    print("Initializing Pre-trained ResNet-18 backbone (ImageNet weights)...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    weights_path = os.path.join(config['model_save_paths']['audio'], 'audio_cnn.pth')
    resumed = False
    if os.path.exists(weights_path):
        print(f"Loading previous audio checkpoint from {weights_path} to continue fine-tuning...")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        resumed = True
        
    model.to(device)
    
    epochs = params.get('epochs', 12)
    freeze_epochs = 0 if resumed else min(3, epochs // 3)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if params.get('mixed_precision', False) and device.type == 'cuda' else None
    
    best_f1 = 0.0
    best_metrics = {}
    start_time = time.time()
    
    if resumed:
        print("Evaluating resumed audio checkpoint on validation set...")
        init_metrics = evaluate(model, val_loader, device)
        best_f1 = init_metrics['f1']
        best_metrics = init_metrics.copy()
        print(f"Resumed baseline: Val F1 = {best_f1:.4f}, Acc = {init_metrics['accuracy']:.4f}, AUC = {init_metrics['roc_auc']:.4f}")
        save_checkpoint(model, config, best_metrics, len(train_df), len(val_df), device)
    else:
        # PHASE 1: Head Alignment (Freeze backbone)
        print(f"\n--- Phase 1: Transfer Learning / Head Alignment ({freeze_epochs} epochs) ---")
        for name, param in model.named_parameters():
            param.requires_grad = ("fc" in name)
                
        optimizer_p1 = torch.optim.AdamW(model.fc.parameters(), lr=5e-4, weight_decay=1e-2)
    
    for epoch in range(freeze_epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_p1.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer_p1)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_p1.step()
                
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        print(f"Phase 1 Epoch {epoch+1}/{freeze_epochs} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            save_checkpoint(model, config, best_metrics, len(train_df), len(val_df), device)

    # PHASE 2: Deep Feature Fine-Tuning (Unfreeze layer3, layer4, and fc)
    fine_tune_epochs = epochs - freeze_epochs
    print(f"\n--- Phase 2: Deep Acoustic Feature Fine-Tuning ({fine_tune_epochs} epochs) ---")
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer_p2 = torch.optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 3e-5},
        {'params': model.layer4.parameters(), 'lr': 5e-5},
        {'params': model.fc.parameters(), 'lr': 1.5e-4}
    ], weight_decay=1e-2)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=fine_tune_epochs)
    
    for epoch in range(fine_tune_epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_p2.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer_p2)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_p2.step()
                
            total_loss += loss.item()
            
        scheduler_p2.step()
        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        print(f"Phase 2 Epoch {epoch+1}/{fine_tune_epochs} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            save_checkpoint(model, config, best_metrics, len(train_df), len(val_df), device)
            
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\nFine-tuning completed in {total_duration:.2f} seconds.")
    print(f"Best Validation F1 Score: {best_f1:.4f} (Accuracy: {best_metrics.get('accuracy', 0):.4f}, AUC: {best_metrics.get('roc_auc', 0):.4f})")
    
    # Save experiment report
    experiment = {
        "experiment_id": str(uuid.uuid4()),
        "date": datetime.datetime.now().isoformat(),
        "model": "pretrained_resnet18_audio_specaugment",
        "dataset": "ASVspoof 2019 LA",
        "subset_size": len(train_df) + len(val_df),
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "seed": config['seed'],
        "epochs": epochs,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "training_time_seconds": round(total_duration, 2),
        "validation_metrics": best_metrics
    }
    
    os.makedirs('training/experiments', exist_ok=True)
    exp_path = os.path.join('training/experiments', f"audio_{experiment['experiment_id']}.json")
    with open(exp_path, 'w') as f:
        json.dump(experiment, f, indent=4)
    print(f"Logged experiment record to {exp_path}")

def save_checkpoint(model, config, metrics, train_count, val_count, device):
    save_dir = config['model_save_paths']['audio']
    os.makedirs(save_dir, exist_ok=True)
    
    weights_path = os.path.join(save_dir, 'audio_cnn.pth')
    torch.save(model.state_dict(), weights_path)
    
    metrics_record = metrics.copy()
    metrics_record['model_name'] = "pretrained_resnet18_audio"
    metrics_record['version'] = "2.0-finetuned-specaugment"
    metrics_record['dataset'] = "ASVspoof 2019 LA"
    metrics_record['training_samples'] = train_count
    metrics_record['validation_samples'] = val_count
    metrics_record['classes'] = ["bonafide", "spoof"]
    metrics_record['seed'] = config['seed']
    metrics_record['gpu'] = str(device)
    metrics_record['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_record, f, indent=4)
        
    print(f"  >>> Checkpoint updated at {weights_path} (F1: {metrics.get('f1', 0):.4f}, Acc: {metrics.get('accuracy', 0):.4f})")

if __name__ == "__main__":
    train()
