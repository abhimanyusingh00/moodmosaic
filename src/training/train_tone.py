import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from src.models.tone_model import ToneClassifier
from src.data.politeness import POLITE_LABELS

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 4
EPOCHS = 5
MAX_LEN = 128
LR = 2e-5

class PolitenessDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
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

    preds = torch.argmax(all_logits, dim=1)

    macro_f1 = f1_score(all_labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    acc = accuracy_score(all_labels.numpy(), preds.numpy())

    loss_fn = nn.CrossEntropyLoss()
    val_loss = loss_fn(all_logits, all_labels).item()

    return val_loss, macro_f1, acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_parquet("data/processed/politeness/train.parquet")
    print("Total politeness examples:", len(df))

    # simple 80/20 split
    split_idx = max(1, int(0.8 * len(df)))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    print("Train size:", len(train_df), "Val size:", len(val_df))

    train_ds = PolitenessDataset(train_df, tokenizer, MAX_LEN)
    val_ds = PolitenessDataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ToneClassifier(MODEL_NAME, num_labels=len(POLITE_LABELS))
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
            print("New best tone model, saving state in memory...")

    save_dir = "experiments/checkpoints/tone/best"
    os.makedirs(save_dir, exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")
    tokenizer.save_pretrained(save_dir)
    print(f"Saved best tone model to {save_dir} with macro F1 = {best_f1:.4f}")

if __name__ == "__main__":
    main()
