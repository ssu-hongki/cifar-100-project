# CIFAR-100 ResNet18 전이학습 기반 학습 코드 (main 함수 구조로 수정)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 학습 함수
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(loader)

# 2. 테스트 함수
def test(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    print(f"Test Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")
    return avg_loss

# 3. 메인 함수 정의
def main():
    # 데이터셋 전처리
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # 데이터셋 불러오기
    train_data = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=2)

    # 모델 정의 (ResNet18 전이학습)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    # 손실 함수, 옵티마이저, 스케줄러
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 학습 반복
    for epoch in range(1, 51):
        print(f"Epoch {epoch}")
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = test(model, test_loader, criterion)
        scheduler.step(val_loss)

    # 모델 저장
    torch.save(model.state_dict(), "resnet18_cifar100.pth")
    print("Model saved to resnet18_cifar100.pth")

# 4. 실행
if __name__ == "__main__":
    main()