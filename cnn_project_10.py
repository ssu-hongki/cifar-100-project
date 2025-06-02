import os
import glob
import tqdm
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import SGD
from PIL import Image

# =========================== 설정 ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 100
SAVE_NAME = "EfficientNet_Clean_Baseline"
EPOCHS = 100
PATIENCE = 10  # 얼리 스탑 기준
os.makedirs("results", exist_ok=True)

# ======================== 모델 정의 =========================
model = efficientnet_b0(pretrained=True)
model.classifier = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

for param in model.parameters():
    param.requires_grad = True

model.to(device)

# ====================== 데이터 전처리 =======================
train_transform = Compose([
    Resize(224),
    RandomCrop(224, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

val_transform = Compose([
    Resize(224),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# ================== CIFAR-100 로딩 + 분할 ==================
full_train = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
train_size = int(len(full_train) * 0.9)
val_size = len(full_train) - train_size
train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 옵티마이저 & 손실함수 =====================
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ===================== 학습 루프 + 얼리 스탑 =====================
best_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optimizer.zero_grad()
        preds = model(data.to(device))
        loss = loss_fn(preds, label.to(device))
        loss.backward()
        optimizer.step()
        iterator.set_description(f"[Train] Epoch {epoch+1} Loss: {loss.item():.4f}")

    # validation accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in val_loader:
            preds = model(data.to(device))
            _, pred_labels = preds.max(1)
            correct += pred_labels.eq(label.to(device)).sum().item()
    acc = correct / len(val_set)
    print(f"\n✅ Validation Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
        torch.save(model.state_dict(), f"results/weight_{SAVE_NAME}.pth")
        print("📌 Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}! Best acc: {best_acc:.4f}")
            break

# ============ 제출용 테스트셋 추론 클래스 ============
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(image_path)
        return filename, image

# =========== 제출용 테스트셋 추론 및 결과 저장 ===========
test_folder = "./Test_Dataset/CImages"
test_transform = Compose([
    Resize(224),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_dataset = TestImageDataset(test_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

output_lines = []
model.eval()
with torch.no_grad():
    for filenames, images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for fname, pred in zip(filenames, predicted):
            num = fname.split('.')[0]
            output_lines.append(f"{num.zfill(4)}, {pred.item():02d}")

with open(f"results/result_{SAVE_NAME}.txt", "w") as f:
    f.write("number, label\n")
    f.write("\n".join(output_lines))

print(f"\n📁 결과 저장 완료: results/result_{SAVE_NAME}.txt")