import os
import yaml
import json
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import uuid
import datetime

class MisinformationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = 1 if self.labels[idx] == 'fake' else 0
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_config():
    with open('training/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
            
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(true_labels, probabilities)
    cm = confusion_matrix(true_labels, predictions)
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

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
    params = config['training_params']['text_misinformation']
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['use_gpu'] else 'cpu')
    print(f"Training Text Misinformation model on {device}")
    
    # Load Manifest
    manifest = pd.read_csv('training/manifests/text_misinformation.csv')
    train_df = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    val_df = manifest[manifest['split'] == 'val'].reset_index(drop=True)
    
    # Combine title and text if title exists
    if 'title' in train_df.columns and 'text' in train_df.columns:
        train_df['combined_text'] = train_df['title'].fillna('') + " " + train_df['text'].fillna('')
        val_df['combined_text'] = val_df['title'].fillna('') + " " + val_df['text'].fillna('')
    else:
        train_df['combined_text'] = train_df['text'].fillna('')
        val_df['combined_text'] = val_df['text'].fillna('')
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model.to(device)
    
    train_dataset = MisinformationDataset(train_df['combined_text'], train_df['label'], tokenizer, params['max_length'])
    val_dataset = MisinformationDataset(val_df['combined_text'], val_df['label'], tokenizer, params['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params['learning_rate']))
    scaler = torch.amp.GradScaler('cuda') if params.get('mixed_precision', False) and device.type == 'cuda' else None
    
    epochs = params['epochs']
    grad_acc_steps = params.get('gradient_accumulation_steps', 1)
    
    best_f1 = 0
    metrics = {}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / grad_acc_steps
                scaler.scale(loss).backward()
                
                if (step + 1) % grad_acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_acc_steps
                loss.backward()
                
                if (step + 1) % grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
            total_loss += loss.item() * grad_acc_steps
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f}")
        
        # Validation
        metrics = evaluate(model, val_loader, device)
        print(f"Validation F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f}")
        
        # Save Checkpoint
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            os.makedirs(config['model_save_paths']['text_misinformation'], exist_ok=True)
            model.save_pretrained(config['model_save_paths']['text_misinformation'])
            tokenizer.save_pretrained(config['model_save_paths']['text_misinformation'])
            
            with open(os.path.join(config['model_save_paths']['text_misinformation'], 'metrics.json'), 'w') as f:
                metrics['model_name'] = "roberta_base_misinformation"
                metrics['version'] = "2.0-finetuned-sanitized"
                metrics['dataset'] = "Fake & Real News Dataset (Sanitized, No Leakage)"
                metrics['training_samples'] = len(train_df)
                metrics['validation_samples'] = len(val_df)
                metrics['classes'] = ["real", "fake"]
                metrics['seed'] = config['seed']
                metrics['gpu'] = str(device)
                metrics['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                json.dump(metrics, f, indent=4)
                
    end_time = time.time()
    
    # Log Experiment
    experiment = {
        "experiment_id": str(uuid.uuid4()),
        "date": datetime.datetime.now().isoformat(),
        "model": "roberta-base",
        "dataset": "Fake & Real News",
        "subset_size": config['subset_sizes']['text_misinformation'],
        "seed": config['seed'],
        "learning_rate": params['learning_rate'],
        "batch_size": params['batch_size'],
        "epochs": epochs,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "training_time_seconds": round(end_time - start_time, 2),
        "validation_metrics": metrics
    }
    
    exp_path = os.path.join('training/experiments', f"text_misinfo_{experiment['experiment_id']}.json")
    with open(exp_path, 'w') as f:
        json.dump(experiment, f, indent=4)

if __name__ == "__main__":
    train()
