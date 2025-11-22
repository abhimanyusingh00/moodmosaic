import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from src.models.personality_model import PersonalityRegressor
from src.data.essays_big5 import BIG5_TRAITS

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-3

class EssaysDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df[[*BIG5_TRAITS]].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        texts = batch["text"]
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(texts, labels=labels)
        loss = out["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            texts = batch["text"]
            labels = batch["labels"].to(device)

            out = model(texts, labels=labels)
            loss = out["loss"]
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_parquet("data/processed/essays_big5/train.parquet")
    print("Total essays:", len(df))

    # tiny dataset, so use same for train and val
    train_df = df
    val_df = df

    train_ds = EssaysDataset(train_df)
    val_ds = EssaysDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = PersonalityRegressor(SBERT_MODEL, hidden_dim=128)
    model.to(device)

    # freeze encoder, train only MLP head
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict().copy()
            print("New best personality model, saving state in memory...")

    save_dir = "experiments/checkpoints/personality"
    os.makedirs(save_dir, exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), f"{save_dir}/best.pt")
    print(f"Saved best personality model to {save_dir}/best.pt with val loss = {best_val:.4f}")

if __name__ == "__main__":
    main()
