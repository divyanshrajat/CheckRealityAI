"""
CheckReality AI - Real GPU Fine-Tuning of Spatial EfficientNet-B0 Backbone
Trains EfficientNet-B0 on cached FaceForensics++ C23 video face crops.
Saves: models/video_deepfake/finetuned_b0_features.pth
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class CachedFramesDataset(Dataset):
    def __init__(self, tensor_path, is_train=False):
        print(f"Loading {tensor_path}...")
        data = torch.load(tensor_path, weights_only=False)
        self.images = data[0] # uint8 [N, 3, 224, 224]
        self.labels = data[1] # long [N]
        self.is_train = is_train
        
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_tx = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
        ])
        print(f"Loaded {len(self.labels)} frames. Train mode: {is_train}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0 # [3, 224, 224]
        if self.is_train:
            img = self.train_tx(img)
        img = self.norm(img)
        lbl = self.labels[idx]
        return img, lbl

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"FINE-TUNING EFFICIENTNET-B0 ON FACEFORENSICS++ C23 FRAMES")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    train_ds = CachedFramesDataset('training/manifests/backbone_train_frames.pt', is_train=True)
    tune_ds = CachedFramesDataset('training/manifests/backbone_tune_frames.pt', is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
    tune_loader = DataLoader(tune_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, 2)
    model = model.to(device)

    epochs = 15
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    class_weights = torch.tensor([2.5, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    scaler = torch.amp.GradScaler('cuda')

    best_tune_loss = float('inf')
    best_tune_acc = 0.0
    out_features_path = 'models/video_deepfake/finetuned_b0_features.pth'
    os.makedirs(os.path.dirname(out_features_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(lbls)
            train_correct += (logits.argmax(dim=1) == lbls).sum().item()
            train_total += len(lbls)

        scheduler.step()
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation on Tune Split
        model.eval()
        tune_loss, tune_correct, tune_total = 0.0, 0, 0
        tune_probs, tune_trues = [], []
        with torch.no_grad():
            for imgs, lbls in tune_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                tune_loss += loss.item() * len(lbls)
                tune_correct += (logits.argmax(dim=1) == lbls).sum().item()
                tune_total += len(lbls)
                probs = torch.softmax(logits, dim=1)[:, 1]
                tune_probs.extend(probs.cpu().numpy())
                tune_trues.extend(lbls.cpu().numpy())

        tune_acc = tune_correct / tune_total
        avg_tune_loss = tune_loss / tune_total
        tune_auc = roc_auc_score(tune_trues, tune_probs)
        tune_f1 = f1_score(tune_trues, [1 if p > 0.5 else 0 for p in tune_probs])
        dur = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({dur:.1f}s) | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Tune Loss: {avg_tune_loss:.4f}, Tune Acc: {tune_acc*100:.2f}%, Tune AUC: {tune_auc:.4f}, Tune F1: {tune_f1:.4f}")

        # Save checkpoint based solely on tune performance
        if avg_tune_loss < best_tune_loss or tune_acc > best_tune_acc:
            best_tune_loss = min(best_tune_loss, avg_tune_loss)
            best_tune_acc = max(best_tune_acc, tune_acc)
            torch.save(model.features.state_dict(), out_features_path)
            print(f"  -> Best model saved to {out_features_path} (Tune Loss: {avg_tune_loss:.4f}, Tune Acc: {tune_acc*100:.2f}%)")

    print("\n" + "=" * 60)
    print(f"BACKBONE FINE-TUNING COMPLETED! Best Tune Acc: {best_tune_acc*100:.2f}%")
    print(f"Saved to: {out_features_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
