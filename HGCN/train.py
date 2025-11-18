import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt  
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from graphs_creation import *
from data import MakeupDataset
from models import MakeupGCN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

NUM_EPOCHS = 50
LEARNING_RATE = 0.0008

print("importing makeup gcn model..")
model = MakeupGCN(in_channels=9, hidden_channels=128, num_layers=4)
model = model.to(device)
print(f"GCN Model: {sum(p.numel() for p in model.parameters()):,} parameters")

print("importing the dataset..")
MAKEUP_DIR = "/content/mt_dataset/all/images/makeupcsv"
NON_MAKEUP_DIR = "/content/mt_dataset/all/images/non-makeupcsv"

dataset = MakeupDataset(
    makeup_csv=MAKEUP_DIR,
    non_csv=NON_MAKEUP_DIR,
    split="train",
    device=device,
    max_samples=None
)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.MSELoss()

train_losses = []
best_loss = float('inf')

print("starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    num_samples = 0

    for idx in tqdm(range(len(dataset)), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        data = dataset[idx]
        if data is None:
            continue

        optimizer.zero_grad()
        pred_colors = model(data)
        n_src = len(data.coords_src)
        loss = criterion(pred_colors[:n_src], data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_samples += 1

    avg_loss = epoch_loss / num_samples
    train_losses.append(avg_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'makeup_gcn_best.pth')
        print("Best model saved")

print("\nGCN training complete!")
