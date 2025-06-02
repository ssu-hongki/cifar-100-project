import os, glob, tqdm, torch, random
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize, AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image

# ==================== 설정 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 100
SAVE_NAME = "EffNetB3_optimized"
EPOCHS = 100
PATIENCE = 10
os.makedirs("results", exist_ok=True)

# ==================== 모델 정의 ====================
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

# ==================== 데이터 전처리 ====================
train_transform = Compose([
    Resize(300),
    RandomCrop(300, padding=8),
    RandomHorizontalFlip(),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==================== 옵티마이저 & 스케줄러 ====================
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==================== CutMix 함수 ====================
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

# ==================== 학습 루프 ====================
import numpy as np
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        data, label = data.to(device), label.to(device)
        
        # CutMix 적용 (50% 확률)
        if np.random.rand() < 0.5:
            data, targets_a, targets_b, lam = cutmix(data, label)
            preds = model(data)
            loss = lam * loss_fn(preds, targets_a) + (1 - lam) * loss_fn(preds, targets_b)
        else:
            preds = model(data)
            loss = loss_fn(preds, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        iterator.set_description(f"[Train] Epoch {epoch+1} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    scheduler.step(epoch)

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"results/weight_{SAVE_NAME}.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("\n⏹️ Early stopping triggered.")
            break

print(f"\n📁 모델 저장 완료: results/weight_{SAVE_NAME}.pth")