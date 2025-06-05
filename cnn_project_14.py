import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

# 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
SAVE_NAME = "YOLOv8X_cls_cifar100"
DATA_DIR = "datasets/cifar100"
TEST_FOLDER = "./Test_Dataset/CImages"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 학습
model = YOLO("yolov8x-cls.pt")
model.train(
    data=DATA_DIR,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    device=device,
    lr0=0.001,
    optimizer='AdamW',
    dropout=0.3,
    label_smoothing=0.1,
    save_period=1,
    save=True,
    verbose=True
)

# 모델 저장
trained_model_dir = model.trainer.save_dir
best_pt_path = os.path.join(trained_model_dir, "weights", "best.pt")
final_pt_path = os.path.join(RESULTS_DIR, f"{SAVE_NAME}.pt")
os.system(f"cp {best_pt_path} {final_pt_path}")

# 테스트셋 클래스
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

# 추론
test_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])
test_dataset = TestImageDataset(TEST_FOLDER, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

inference_model = YOLO(final_pt_path)
output_lines = []

with torch.no_grad():
    for filenames, images in tqdm(test_loader, desc="🔍 Inference"):
        images = images.to(device)
        results = inference_model(images, verbose=False)
        pred_classes = torch.argmax(results[0], dim=1).cpu().numpy()
        for fname, pred in zip(filenames, pred_classes):
            num = fname.split('.')[0]
            output_lines.append(f"{num.zfill(4)}, {pred:02d}")

# 저장
txt_path = os.path.join(RESULTS_DIR, f"{SAVE_NAME}.txt")
with open(txt_path, "w") as f:
    f.write("number, label\n")
    f.write("\n".join(output_lines))

print(f"\n✅ 결과 저장 완료: {txt_path}")
print(f"✅ 가중치 저장 완료: {final_pt_path}")