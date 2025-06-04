# evaluate_effnet_test.py

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3
import tqdm

# ============ 설정 ============
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100
BATCH_SIZE = 64
WEIGHT_PATH = "results/weight_가반3조_0604_1800.pth"  # 저장된 모델 경로

# ============ 모델 정의 ============
model = efficientnet_b3(pretrained=False)  # pretrained=False 꼭 필요!
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device)
model.eval()

# ============ CIFAR-100 테스트셋 로딩 ============
transform = Compose([
    Resize(300),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============ 정확도 평가 ============
correct = 0
total = 0
with torch.no_grad():
    for data, labels in tqdm.tqdm(test_loader, desc="Evaluating"):
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total * 100
print(f"\n✅ CIFAR-100 Test Accuracy: {test_acc:.2f}%")