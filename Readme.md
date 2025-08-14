## 🐦 Classification d’espèces d’oiseaux (YOLOv8n‑cls, ROCm, Ubuntu 24.04)

Modèle de classification d’images basé sur **YOLOv8n‑cls** pour reconnaître l’espèce d’un oiseau. Le projet est optimisé pour **Ubuntu 24.04** et **GPU AMD (ROCm)**. L’entraînement reprend automatiquement depuis le dernier modèle personnalisé s’il est présent.

---

## ✅ En Résumé

- **Jeu de données attendu**: `CUB_200_2011/images/<nom_espece>/*.jpg`
- **Source du dataset**: [CUB‑200‑2011 (Caltech)](https://data.caltech.edu/records/65de6-vp158)
- **Préparation automatique**: création de `birds_yolo_cls/` avec split **train/val (80/20)**
- **Tâche**: classification uniquement (pas de bounding boxes)
- **Reprise**: si `birds_yolov8n-cls.pt` existe, l’entraînement repart de ce modèle
- **Sauvegarde**: le meilleur modèle est copié en `birds_yolov8n-cls.pt`
- **Compatibilité**: PyTorch ROCm (GPU AMD)

---

## ⚙️ Installation (ROCm / AMD)

Important: utilisez les roues ROCm officielles pour PyTorch.

```bash
python -m venv venv
source venv/bin/activate 

pip install --upgrade pip
pip install -r requirements.txt
```

### Vérifier la détection ROCm
```bash
python test_rocm.py
```
Sur AMD/ROCm, `torch.version.hip` doit être défini. Ultralytics peut afficher un log du type:
`CUDA:0 (Radeon RX 7900 XTX, 24560MiB)`

---

## 📦 Organisation des données

```
CUB_200_2011/
└─ images/
   ├─ species_1/ *.jpg
   ├─ species_2/ *.jpg
   └─ ... (200 espèces)
```

Conseils:
- Un dossier par espèce.
- Format `.jpg` attendu (adaptez le script pour d’autres extensions si besoin).

---

## 🚀 Utilisation

### 1) Entraîner (ou reprendre)
```bash
python train.py
```
- Premier lancement: charge `yolov8n-cls.pt` (pré‑entraîné ImageNet)
- Relance: si `birds_yolov8n-cls.pt` est présent, reprise à partir de ce modèle

Comportement du script:
- Prépare `birds_yolo_cls/{train,val}` (recréé à chaque exécution)
- Entraîne sur ROCm (`device=0`)
- Sauvegarde le meilleur modèle sous `runs/classify/train/weights/best.pt` et le copie en `birds_yolov8n-cls.pt`

### 2) Évaluer (Top‑1 / Top‑5)
```bash
python eval.py
```
- Utilise par défaut `runs/classify/train/weights/best.pt` (modifiez `MODEL_PATH` dans `eval.py` si nécessaire)
- Mesures:
  - Top‑1: la meilleure prédiction correspond à la bonne classe
  - Top‑5: la bonne classe est dans les 5 meilleures prédictions

### 3) Tester sur une image
Dans `eval.py`, définissez:
```python
IMG_TEST = "path/to/your_image.jpg"
```
Puis exécutez:
```bash
python eval.py
```
Le script affichera les probabilités par classe pour l’image.

---

## 🧠 Paramètres par défaut (dans `train.py`)

- `IMG_SIZE = 224`
- `EPOCHS = 50`
- `BATCH_SIZE = 32`
- Optimiseur: **AdamW**, `lr0 = 1e-3`, `patience = 10`
- Split **train/val = 80/20** (recréé à partir de `CUB_200_2011/images`)

Vous pouvez ajuster ces valeurs directement dans `train.py`.

---

## 📂 Arborescence du projet

```
PyBirds/
├─ train.py                  # Prépare le dataset + entraîne YOLOv8n‑cls (reprend si modèle existe)
├─ eval.py                   # Évalue (Top‑1 / Top‑5)
├─ requirements.txt          
├─ CUB_200_2011/
│  └─ images/
│     ├─ species_1/ *.jpg
│     ├─ species_2/ *.jpg
│     └─ ... (200 espèces)
└─ birds_yolo_cls/           # GÉNÉRÉ (train/val) au premier lancement
   ├─ train/
   └─ val/
```

---

## 🩹 Dépannage

- Démarre sur le modèle de base au lieu de reprendre:
  - Vérifiez la présence de `birds_yolov8n-cls.pt` avant `python train.py`
  - Le script affiche l’un des messages suivants:
    - `📂 Reprise de l'entraînement depuis birds_yolov8n-cls.pt`
    - `🆕 Nouvel entraînement depuis yolov8n-cls.pt`

- Problèmes d’installation PyTorch ROCm:
  - Conservez l’index ROCm dans `requirements.txt`
  - Ou installez avec `pip install ... --index-url https://download.pytorch.org/whl/rocm6.1`

- Mémoire GPU saturée:
  - Réduisez `BATCH_SIZE` (par ex. `16` ou `8`)

- Images non `.jpg`:
  - Modifiez `prepare_dataset()` pour inclure d’autres extensions:
    ```python
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
    ```

---

## 📜 Licence

Usage libre pour projets **éducatifs** et **expérimentaux**. Vérifiez les licences des datasets utilisés (ex. CUB‑200‑2011).