# efficientNet b3 기본 모델 기법 거의 추가 안한 버젼
import os
import glob
import tqdm
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image

# =========================== 설정 ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 100
SAVE_NAME = "EffNetB3_fulltrain"
EPOCHS = 100
PATIENCE = 10
os.makedirs("results", exist_ok=True)

# ======================== 모델 정의 =========================
model = efficientnet_b3(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

# ✅ 전체 파라미터 학습 (프리징 없음)
for param in model.parameters():
    param.requires_grad = True

model.to(device)

# ====================== 데이터 전처리 =======================
train_transform = Compose([
    Resize(300),
    RandomCrop(300, padding=8),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# ================ CIFAR-100 전체를 학습에만 사용 ================
train_dataset = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 옵티마이저 & 손실함수 =====================
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# ===================== 학습 루프 =====================
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optimizer.zero_grad()
        preds = model(data.to(device))
        loss = loss_fn(preds, label.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iterator.set_description(f"[Train] Epoch {epoch+1} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

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