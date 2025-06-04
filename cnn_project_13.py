from ultralytics import YOLO
import os
import shutil
import sys

# 하이퍼파라미터 설정
model_path = "yolov8s-cls.pt"
data_dir = "datasets/cifar100"  # train/val 폴더 구조
imgsz = 224
epochs = 200
batch = 256
device = "cuda"
save_name = f"yolov8s_cls_cifar100_e{epochs}_b{batch}_img{imgsz}"
log_txt = f"{save_name}.txt"

# 표준 출력 로그 파일에 저장
sys.stdout = open(log_txt, 'w')

print("✅ YOLOv8s 분류 전이학습 시작")
print(f"모델: {model_path}")
print(f"데이터셋: {data_dir}")
print(f"하이퍼파라미터 - imgsz: {imgsz}, epochs: {epochs}, batch: {batch}")

# 모델 로드
model = YOLO(model_path)

# 학습
model.train(
    data=data_dir,
    imgsz=imgsz,
    epochs=epochs,
    batch=batch,
    device=device,
    patience=epochs + 1,  # patience를 epochs보다 크게 설정해서 얼리스타핑 무효화
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=5,
    label_smoothing=0.1,
    mixup=0.2,
    cutmix=0.3,
    dropout=0.2
)

# 학습 완료 후 best.pt → .pth 저장
trained_model_dir = model.trainer.save_dir  # runs/classify/train/ 등
src = os.path.join(trained_model_dir, "weights", "best.pt")
dst = os.path.join("saved_models", f"{save_name}.pth")
os.makedirs("saved_models", exist_ok=True)
shutil.copy(src, dst)

print(f"\n✅ 학습 완료 및 모델 저장됨: {dst}")
print(f"📄 로그 저장 완료: {log_txt}")

# stdout 다시 원래대로
sys.stdout.close()
sys.stdout = sys.__stdout__