import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from src.data.goemotions import GOEMO_LABELS
from src.models.emotion_model import EmotionClassifier

MODEL_NAME = "roberta-base"
BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 128
LR = 2e-5

class GoEmoDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float),
        }
        return item

def load_datasets():
    train_df = pd.read_parquet("data/processed/goemotions/train.parquet")
    val_df = pd.read_parquet("data/processed/goemotions/validation.parquet")
    return train_df, val_df

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = outputs["logits"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits)
    preds = (probs > 0.5).int()

    macro_f1 = f1_score(all_labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    acc = accuracy_score(all_labels.numpy(), preds.numpy())

    loss_fn = nn.BCEWithLogitsLoss()
    val_loss = loss_fn(all_logits, all_labels).item()

    return val_loss, macro_f1, acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df, val_df = load_datasets()
    print("Train size:", len(train_df), "Val size:", len(val_df))

    train_dataset = GoEmoDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = GoEmoDataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = EmotionClassifier(MODEL_NAME, num_labels=len(GOEMO_LABELS))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, macro_f1, acc = eval_epoch(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print(f"Accuracy:   {acc:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = model.state_dict().copy()
            print("New best model, saving state in memory...")

    # Save best model and tokenizer
    save_dir = "experiments/checkpoints/emotion/best"
    import os
    os.makedirs(save_dir, exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")
    tokenizer.save_pretrained(save_dir)
    print(f"Saved best model to {save_dir} with macro F1 = {best_f1:.4f}")

if __name__ == "__main__":
    main()
