import os, glob, torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# ============ 설정 ============
device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100
SAVE_NAME = "EffNetB3_optimized"
WEIGHT_PATH = f"results/weight_{SAVE_NAME}.pth"
TEST_FOLDER = "./Test_Dataset/CImages"
TXT_OUTPUT = f"results/result_{SAVE_NAME}.txt"

# ============ 모델 정의 ============
model = efficientnet_b3(pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.classifier[1].in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, NUM_CLASSES)
)
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device)
model.eval()

# ============ 테스트셋 클래스 ============
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

# ============ 전처리 및 로더 ============
test_transform = Compose([
    Resize(300),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_dataset = TestImageDataset(TEST_FOLDER, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============ 추론 및 저장 ============
output_lines = []
with torch.no_grad():
    for filenames, images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for fname, pred in zip(filenames, predicted):
            num = fname.split('.')[0]
            output_lines.append(f"{num.zfill(4)}, {pred.item():02d}")

os.makedirs("results", exist_ok=True)
with open(TXT_OUTPUT, "w") as f:
    f.write("number, label\n")
    f.write("\n".join(output_lines))

print(f"\n📁 .txt 추론 결과 저장 완료: {TXT_OUTPUT}")