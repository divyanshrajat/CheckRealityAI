import os
import yaml
import json
import time
import uuid
import datetime
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ImageDeepfakeDataset(Dataset):
    def __init__(self, filepaths, labels, base_dir="Datasets/image/real_vs_fake/real-vs-fake", transform=None):
        self.filepaths = filepaths.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path_val = str(self.filepaths.iloc[idx])
        if not os.path.isabs(path_val):
            full_path = os.path.join(self.base_dir, path_val)
        else:
            full_path = path_val
            
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
        if self.transform:
            image = self.transform(image)
            
        label_val = str(self.labels.iloc[idx]).strip().lower()
        # Binary target: 1 = Fake, 0 = Real
        label = 1 if (label_val == 'fake' or label_val == '0') else 0
        return image, torch.tensor(label, dtype=torch.long)

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
    params = config['training_params']['image']
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['use_gpu'] else 'cpu')
    print("=== CheckReality Image Deepfake Fine-Tuning ===")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Load Verified Manifest (Guaranteed Zero Data Leakage)
    manifest_path = 'training/manifests/image_deepfake.csv'
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run create_subsets.py first.")
        
    manifest = pd.read_csv(manifest_path)
    train_df = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    val_df = manifest[manifest['split'] == 'val'].reset_index(drop=True)
    
    # Verify zero data leakage
    leak_check = set(train_df['path']).intersection(set(val_df['path']))
    assert len(leak_check) == 0, f"DATA LEAKAGE DETECTED: {len(leak_check)} overlapping paths!"
    print(f"Data Leakage Verification PASSED: 0 overlapping images between train and val.")
    
    # Use label_str if available for unambiguous mapping
    train_label_col = 'label_str' if 'label_str' in train_df.columns else 'label'
    val_label_col = 'label_str' if 'label_str' in val_df.columns else 'label'
    
    print(f"Training dataset: {len(train_df)} samples ({train_df[train_label_col].value_counts().to_dict()})")
    print(f"Validation dataset: {len(val_df)} samples ({val_df[val_label_col].value_counts().to_dict()})")
    
    img_size = params.get('image_size', 224)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageDeepfakeDataset(train_df['path'], train_df[train_label_col], transform=train_transform)
    val_dataset = ImageDeepfakeDataset(val_df['path'], val_df[val_label_col], transform=val_transform)
    
    batch_size = params.get('batch_size', 16)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize Pre-trained EfficientNet-B3
    print("Initializing EfficientNet-B3 backbone...")
    weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    model = models.efficientnet_b3(weights=weights)
    
    # Replace classifier head for 2 classes (Real=0 vs Fake=1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    weights_path = os.path.join(config['model_save_paths']['image'], 'efficientnet_b3.pth')
    resumed = False
    if os.path.exists(weights_path):
        print(f"Loading previous checkpoint from {weights_path} to continue improvement...")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        resumed = True
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    best_metrics = {}
    start_time = time.time()
    
    epochs_p1 = params.get('epochs_phase1', 2)
    best_acc = 0.0
    if resumed:
        print("Evaluating resumed checkpoint on validation set...")
        init_metrics = evaluate(model, val_loader, device)
        best_f1 = init_metrics['f1']
        best_acc = init_metrics['accuracy']
        best_metrics = init_metrics.copy()
        print(f"Resumed baseline: Val F1 = {best_f1:.4f}, Acc = {init_metrics['accuracy']:.4f}, AUC = {init_metrics['roc_auc']:.4f}")
        # Always save immediately to refresh metadata timestamp
        save_checkpoint(model, config, best_metrics, len(train_df), len(val_df), device)
    else:
        # PHASE 1: Head Alignment (Freeze backbone)
        print(f"\n--- Phase 1: Classifier Head Alignment ({epochs_p1} epochs) ---")
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        optimizer_p1 = torch.optim.AdamW(model.classifier.parameters(), lr=float(params.get('learning_rate_phase1', 1e-3)), weight_decay=1e-2)
        
        for epoch in range(epochs_p1):
            model.train()
            total_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_p1.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_p1.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / max(1, len(train_loader))
            val_metrics = evaluate(model, val_loader, device)
            print(f"Phase 1 Epoch {epoch+1}/{epochs_p1} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                save_checkpoint(model, config, best_metrics, len(train_df), len(val_df), device)

    # PHASE 2: Ultra Deep Feature Fine-Tuning (Unfreeze last 6 convolutional blocks)
    epochs_p2 = params.get('epochs_phase2', 3)
    print(f"\n--- Phase 2: Ultra Deep Feature Fine-Tuning ({epochs_p2} epochs) ---")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-6:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    optimizer_p2 = torch.optim.AdamW([
        {'params': model.features[-6:].parameters(), 'lr': 3.5e-5},
        {'params': model.classifier.parameters(), 'lr': 8e-5}
    ], weight_decay=1e-2)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=epochs_p2, eta_min=1e-6)
    
    for epoch in range(epochs_p2):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_p2.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_p2.step()
            total_loss += loss.item()
            
        scheduler_p2.step()
        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device)
        print(f"Phase 2 Epoch {epoch+1}/{epochs_p2} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")
        
        if val_metrics['f1'] >= best_f1 or val_metrics['accuracy'] >= best_acc:
            if val_metrics['f1'] > best_f1: best_f1 = val_metrics['f1']
            if val_metrics['accuracy'] > best_acc: best_acc = val_metrics['accuracy']
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
        "model": "efficientnet_b3_deepfake_enhanced",
        "dataset": "140k Real and Fake Faces",
        "subset_size": len(train_df) + len(val_df),
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "seed": config['seed'],
        "epochs": epochs_p2,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "training_time_seconds": round(total_duration, 2),
        "validation_metrics": best_metrics
    }
    
    os.makedirs('training/experiments', exist_ok=True)
    exp_path = os.path.join('training/experiments', f"image_{experiment['experiment_id']}.json")
    with open(exp_path, 'w') as f:
        json.dump(experiment, f, indent=4)
    print(f"Logged experiment record to {exp_path}")

def save_checkpoint(model, config, metrics, train_count, val_count, device):
    save_dir = config['model_save_paths']['image']
    os.makedirs(save_dir, exist_ok=True)
    
    weights_path = os.path.join(save_dir, 'efficientnet_b3.pth')
    torch.save(model.state_dict(), weights_path)
    
    metrics_record = metrics.copy()
    metrics_record['model_name'] = "efficientnet_b3_deepfake"
    metrics_record['version'] = "3.0-ultra-deepfake-99"
    metrics_record['dataset'] = "140k Real and Fake Faces"
    metrics_record['training_samples'] = train_count
    metrics_record['validation_samples'] = val_count
    metrics_record['classes'] = ["real", "fake"]
    metrics_record['seed'] = config['seed']
    metrics_record['gpu'] = str(device)
    metrics_record['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_record, f, indent=4)
        
    print(f"  >>> Checkpoint updated at {weights_path} (F1: {metrics.get('f1', 0):.4f}, Acc: {metrics.get('accuracy', 0):.4f})")

if __name__ == "__main__":
    train()
