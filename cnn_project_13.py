import os
import glob
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ================ 설정 ================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 100
EPOCHS = 100
SAVE_NAME = "YOLOv8_customhead_0604"
DATA_DIR = "datasets/cifar100"
TEST_DIR = "./Test_Dataset/CImages"
os.makedirs("results", exist_ok=True)

# ================ 모델 정의 (분류기 커스터마이징) ================
model = YOLO("yolov8s-cls.pt")
model.model.model[-1] = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)
model.model.to(device)

# ================ 데이터 전처리 ================
train_transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])
val_transform = train_transform  # 여기선 같게 씀

train_dataset = DatasetFolder = torch.utils.data.dataset.DatasetFolder(
    os.path.join(DATA_DIR, "train"),
    loader=lambda path: Image.open(path).convert("RGB"),
    extensions=("png", "jpg"),
    transform=train_transform
)
val_dataset = DatasetFolder = torch.utils.data.dataset.DatasetFolder(
    os.path.join(DATA_DIR, "val"),
    loader=lambda path: Image.open(path).convert("RGB"),
    extensions=("png", "jpg"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================ 손실함수, 옵티마이저 ================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3, weight_decay=1e-4)

BEST_ACC = 0.0

# ================ 학습 루프 ================
for epoch in range(EPOCHS):
    model.model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}]", leave=False)
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
    print(f"📚 Epoch {epoch+1} - Train Accuracy: {train_acc:.2f}%")

    # 검증
    model.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model.model(x)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total * 100
    print(f"✅ Epoch {epoch+1} - Validation Accuracy: {val_acc:.2f}%")

    # best 저장
    if val_acc > BEST_ACC:
        BEST_ACC = val_acc
        torch.save(model.model.state_dict(), f"results/weight_{SAVE_NAME}.pth")
        print(f"💾 Best model 저장됨: results/weight_{SAVE_NAME}.pth")

# ================ 제출용 테스트셋 추론 ================
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

# 테스트셋 추론 및 결과 저장
test_transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])
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

with open(f"results/result_{SAVE_NAME}.txt", "w") as f:
    f.write("number, label\n")
    f.write("\n".join(output_lines))

print(f"\n📁 결과 저장 완료: results/result_{SAVE_NAME}.txt")