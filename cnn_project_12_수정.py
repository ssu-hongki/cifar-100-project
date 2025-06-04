import os, glob, tqdm, torch, random
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b3
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize, AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 100
SAVE_NAME = "가반3조_0604_1800"
EPOCHS = 100
PATIENCE = 10
VAL_RATIO = 0.1  
os.makedirs("results", exist_ok=True)

model = efficientnet_b3(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)
for param in model.parameters():
    param.requires_grad = True
model.to(device)

train_transform = Compose([
    Resize(300),
    RandomCrop(300, padding=8),
    RandomHorizontalFlip(),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
val_transform = Compose([
    Resize(300),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

full_dataset = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
train_size = int(len(full_dataset) * (1 - VAL_RATIO))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

def cutmix(data, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(data.size()[0]).to(device)
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    iterator = tqdm.tqdm(train_loader)

    for data, label in iterator:
        data, label = data.to(device), label.to(device)

        if np.random.rand() < 0.5:
            data, targets_a, targets_b, lam = cutmix(data, label)
            preds = model(data)
            loss = lam * loss_fn(preds, targets_a) + (1 - lam) * loss_fn(preds, targets_b)
        else:
            preds = model(data)
            loss = loss_fn(preds, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        _, pred_labels = preds.max(1)
        correct += (pred_labels == label).sum().item()
        total += label.size(0)
        iterator.set_description(f"[Train] Epoch {epoch+1} Loss: {loss.item():.4f}")

    train_acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)
    scheduler.step(epoch)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            preds = model(data)
            _, pred_labels = preds.max(1)
            val_correct += (pred_labels == label).sum().item()
            val_total += label.size(0)
    val_acc = val_correct / val_total * 100

    print(f"\nEpoch {epoch+1}: Train Acc = {train_acc:.2f}% | Val Acc = {val_acc:.2f}%")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"results/weight_{SAVE_NAME}.pth")
        print(f"Best model 저장됨: results/weight_{SAVE_NAME}.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

print(f"\n학습 완료. 최종 모델 저장 위치: results/weight_{SAVE_NAME}.pth")