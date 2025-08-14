import os
import random
import shutil
from ultralytics import YOLO
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
DATASET_DIR = Path("./CUB_200_2011/images")  # Dossier contenant les images classées par dossier
OUTPUT_DIR = Path("./birds_yolo_cls")        # Dataset YOLOv8 format (sera recréé)
TRAIN_SPLIT = 0.8
IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 50

# Noms des modèles
CUSTOM_MODEL_NAME = "birds_yolov8n-cls.pt"   # Ton modèle entraîné
PRETRAINED_MODEL = "yolov8n-cls.pt"          # Modèle pré-entraîné YOLOv8n classification
# -------------------------------


def prepare_dataset():
    """
    Crée un dataset train/val au format YOLOv8 classification
    en copiant les images depuis DATASET_DIR.
    """
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"

    # Supprimer l'ancien dataset
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Parcourir les classes
    for class_dir in DATASET_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            random.shuffle(images)
            split_idx = int(len(images) * TRAIN_SPLIT)

            train_class_dir = train_dir / class_dir.name
            val_class_dir = val_dir / class_dir.name
            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)

            # Copier les images
            for img in images[:split_idx]:
                shutil.copy(img, train_class_dir / img.name)
            for img in images[split_idx:]:
                shutil.copy(img, val_class_dir / img.name)

    print(f"✅ Dataset préparé dans {OUTPUT_DIR}")


def train_model():
    """
    Entraîne YOLOv8n-cls avec ROCm et reprend si un modèle custom existe déjà.
    """
    # Choix du modèle de départ
    if Path(CUSTOM_MODEL_NAME).exists():
        print(f"📂 Reprise de l'entraînement depuis {CUSTOM_MODEL_NAME}")
        model = YOLO(CUSTOM_MODEL_NAME)
    else:
        print(f"🆕 Nouvel entraînement depuis {PRETRAINED_MODEL}")
        model = YOLO(PRETRAINED_MODEL)

    # Entraînement
    model.train(
        data=str(OUTPUT_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,  # GPU ROCm
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,
        patience=10,
        verbose=True
    )

    # Sauvegarde du meilleur modèle
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, CUSTOM_MODEL_NAME)
        print(f"💾 Modèle sauvegardé sous {CUSTOM_MODEL_NAME}")


if __name__ == "__main__":
    random.seed(42)
    prepare_dataset()
    train_model()
