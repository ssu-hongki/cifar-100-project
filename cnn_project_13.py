import os, glob, tqdm, torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import CIFAR100, ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from ultralytics import YOLO

# ============ 설정 ============
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100
EPOCHS = 100
BATCH_SIZE = 64
SAVE_NAME = "YOLOv8_customhead_fixed"
DATA_DIR = "./datasets/cifar100"
TEST_DIR = "./Test_Dataset/CImages"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============ CIFAR-100 폴더 변환 ============
def save_images(images, labels, classes, root_dir):
    for idx, (img_arr, label) in enumerate(tqdm.tqdm(zip(images, labels), total=len(images))):
        class_name = classes[label]
        class_dir = os.path.join(root_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img = Image.fromarray(img_arr)
        img.save(os.path.join(class_dir, f"{idx:05}.png"))

def convert_cifar100_to_imagefolder():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("✅ CIFAR-100 폴더 구조 이미 존재함")
        return
    print("🔄 CIFAR-100을 ImageFolder 구조로 변환 중...")
    train_set = CIFAR100(root="./", train=True, download=True)
    test_set = CIFAR100(root="./", train=False, download=True)
    save_images(train_set.data, train_set.targets, train_set.classes, train_dir)
    save_images(test_set.data, test_set.targets, test_set.classes, val_dir)
    print("✅ 변환 완료")

convert_cifar100_to_imagefolder()

# ============ 데이터 로딩 ============
transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
train_dataset = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============ YOLO 모델 로드 + FC 헤드만 수정 ============
model = YOLO("yolov8s-cls.pt")

# ✅ Sequential 전체를 덮지 않고 마지막 레이어만 정확히 교체
if isinstance(model.model.model, nn.Sequential):
    last_in = model.model.model[-1].in_features if isinstance(model.model.model[-1], nn.Linear) else 1024
    model.model.model[-1] = nn.Sequential(
        nn.Linear(last_in, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
else:
    raise ValueError("model.model.model이 Sequential 구조가 아닙니다.")

model.model.to(device)

# ============ 학습 설정 ============
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3, weight_decay=1e-4)
best_acc = 0.0

# ============ 학습 루프 ============
for epoch in range(EPOCHS):
    model.model.train()
    correct, total, total_loss = 0, 0, 0

    loop = tqdm.tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        outputs = model.model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), acc=correct/total*100)

    train_acc = correct / total * 100

    # ============ 검증 ============
    model.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model.model(x)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    val_acc = correct / total * 100

    print(f"\n📊 Epoch {epoch+1}: Train Acc = {train_acc:.2f}% | Val Acc = {val_acc:.2f}%")

    # ============ best 저장 ============
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.model.state_dict(), f"{SAVE_DIR}/weight_{SAVE_NAME}.pth")
        print(f"💾 Best model 저장됨: {SAVE_DIR}/weight_{SAVE_NAME}.pth")

# ============ 테스트셋 추론 Dataset 클래스 ============
class TestImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(image_path)
        return filename, image

# ============ 테스트셋 추론 및 결과 저장 ============
test_transform = transform
test_dataset = TestImageDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

output_lines = []
model.model.eval()
with torch.no_grad():
    for filenames, images in test_loader:
        images = images.to(device)
        outputs = model.model(images)
        _, predicted = torch.max(outputs, 1)
        for fname, pred in zip(filenames, predicted):
            num = fname.split('.')[0]
            output_lines.append(f"{num.zfill(4)}, {pred.item():02d}")

with open(f"{SAVE_DIR}/result_{SAVE_NAME}.txt", "w") as f:
    f.write("number, label\n")
    f.write("\n".join(output_lines))

print(f"\n✅ 테스트셋 추론 결과 저장 완료: {SAVE_DIR}/result_{SAVE_NAME}.txt")