import os, glob, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# ===== 설정 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 64
SAVE_NAME = "YOLOv8X_cls_cifar100"  # 저장된 .pt 이름 기준
TEST_FOLDER = "./Test_Dataset/CImages"
RESULTS_DIR = "results"
PT_PATH = f"{RESULTS_DIR}/{SAVE_NAME}.pt"
TXT_PATH = f"{RESULTS_DIR}/{SAVE_NAME}.txt"

# ===== 테스트셋 로더 클래스 =====
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

# ===== 전처리 정의 =====
test_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])
test_dataset = TestImageDataset(TEST_FOLDER, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== 저장된 모델 로딩 =====
model = YOLO(PT_PATH)
output_lines = []

# ===== 추론 및 저장 =====
with torch.no_grad():
    for filenames, images in tqdm(test_loader, desc="🔍 Inference"):
        images = images.to(device)
        results = model(images, verbose=False)

        for fname, result in zip(filenames, results):
            pred = int(result.probs.data.argmax())  # ← 수정된 라벨 추출
            num = fname.split('.')[0]
            output_lines.append(f"{num.zfill(4)}, {pred:02d}")

# ===== txt 파일 저장 =====
with open(TXT_PATH, "w") as f:
    f.write("number, label\n")
    f.write("\n".join(output_lines))

print(f"\n✅ 결과 저장 완료: {TXT_PATH}")