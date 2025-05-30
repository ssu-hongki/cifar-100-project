import os
import glob
import tqdm
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomCrop, Normalize, ColorJitter, RandomRotation
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from PIL import Image

# =========================== 설정 ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_CLASSES = 100
SAVE_NAME = "Gaban3jo_0602_1600"
os.makedirs("results", exist_ok=True)

# ======================== 모델 정의 =========================
model = efficientnet_b0(pretrained=True)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)
model.to(device)

# ====================== 데이터 전처리 =======================
train_transform = Compose([
    Resize(224),
    RandomCrop(224, padding=4),
    RandomHorizontalFlip(),
    RandomRotation(15),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
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
train_set, val_set = random_split(full_train, [train_size, val_size])
val_set.dataset.transform = val_transform  # 검증셋은 증강 X

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ===================== 최적화 설정 =====================
for name, param in model.named_parameters():
    param.requires_grad = ("classifier" in name)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ===================== 전이학습 단계 =====================
for epoch in range(100):
    model.train()
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optimizer.zero_grad()
        preds = model(data.to(device))
        loss = loss_fn(preds, label.to(device))
        loss.backward()
        optimizer.step()
        iterator.set_description(f"[Transfer] Epoch {epoch+1} Loss: {loss.item():.4f}")

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

# ===================== 모델 저장 =====================
torch.save(model.state_dict(), f"results/weight_{SAVE_NAME}.pth")

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