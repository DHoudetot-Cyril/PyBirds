from ultralytics import YOLO
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "runs/classify/train/weights/best.pt"  # chemin vers ton modèle entraîné
DATASET_DIR = Path("./birds_yolo_cls/val")  # dossier de validation
IMG_TEST = None  # exemple: "test_images/oiseau.jpg"
# -------------------------------

def evaluate_model():
    """Évalue la précision du modèle sur le dataset de validation."""
    model = YOLO(MODEL_PATH)

    print("🔍 Évaluation sur le dataset de validation...")
    metrics = model.val(data=str(DATASET_DIR), split="val", device=0)  # ROCm GPU

    print("\n📊 Résultats:")
    print(f"Top-1 Accuracy : {metrics.top1:.4f}")
    print(f"Top-5 Accuracy : {metrics.top5:.4f}")

def test_on_image():
    """Teste le modèle sur une image ou un dossier."""
    if IMG_TEST is None:
        return

    model = YOLO(MODEL_PATH)
    results = model.predict(source=IMG_TEST, imgsz=224, device=0)

    for r in results:
        print(f"\n📸 Image : {r.path}")
        for label, prob in zip(r.names.values(), r.probs.data.tolist()):
            print(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    evaluate_model()
    test_on_image()
