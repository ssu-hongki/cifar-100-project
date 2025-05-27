import tqdm
import torch
import torch.nn as nn
from torchsummary import summary

from torchvision.models.resnet import resnet18
from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 사전 학습된 모델 준비
model = resnet18(pretrained=True)
num_output = 100  # CIFAR-100 클래스 수
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_output)
model.to(device)
print(model)

# 모델 요약 출력
summary(model, input_size=(3, 224, 224))

# 데이터 전처리 및 증강
transforms = Compose([
   Resize(224),
   RandomCrop((224, 224), padding=4),
   RandomHorizontalFlip(p=0.5),
   ToTensor(),
   Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# CIFAR-100 데이터셋 로드
training_data = CIFAR100(root="./data", train=True, download=True, transform=transforms)
test_data = CIFAR100(root="./data", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 파라미터 freezing 및 학습 대상 설정
params_name_to_update = ['fc.weight', 'fc.bias']
params_to_update = []
for name, param in model.named_parameters():
    if name in params_name_to_update:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

# 옵티마이저 설정
lr = 1e-4
optimizer = Adam(params=params_to_update, lr=lr)

# 학습 루프
epochs = 5
for epoch in range(epochs):
    model.train()
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        optimizer.zero_grad()
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optimizer.step()
        iterator.set_description(f"epoch:{epoch+1:02d} loss:{loss.item():.4f}")

# 모델 저장
torch.save(model.state_dict(), "CIFAR100_pretrained_ResNet.pth")

# 모델 로드 및 평가
model.load_state_dict(torch.load("CIFAR100_pretrained_ResNet.pth", map_location=device))
model.eval()
num_corr = 0
with torch.no_grad():
    for data, label in test_loader:
        output = model(data.to(device))
        _, preds = output.data.max(1)
        corr = preds.eq(label.to(device)).sum().item()
        num_corr += corr

accuracy = num_corr / len(test_data) * 100
print(f"Test Accuracy: {accuracy:.2f}%")