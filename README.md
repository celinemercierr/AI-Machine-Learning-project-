## Dataset

**Food Recognition 2022** (Kaggle) — ingredients with bounding boxes in COCO format, converted to YOLO format for training.

- Training set: ~24,000 images
- Validation set: ~1,200 images  
- Format: YOLO (images/ + labels/)

Dataset is stored in Google Drive, not committed to the repo.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/celinemercierr/AI-Machine-Learning-project-.git
cd AI-Machine-Learning-project-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset (Google Colab)
```python
import kagglehub
path = kagglehub.dataset_download("sainikhileshreddy/food-recognition-2022")
```

## Training
```bash
python src/train.py --data data/processed/dataset.yaml --epochs 30 --batch 16 --device cuda
```

For Colab free tier:
```bash
python src/train.py --epochs 5 --batch 8 --device cuda
```

## Inference
```bash
python src/infer.py --source path/to/fridge.jpg --weights outputs/checkpoints/best.pt --conf 0.4
```

## Demo
```bash
python app.py
```

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 |
|-------|---------|--------------|
| YOLOv8n baseline | — | — |
| YOLOv8n fine-tuned | — | — |

*To be filled after training.*

## Team

| Name | Role |
|------|------|
| Maria | ML Engineer + Data Engineer |
| Celine | ML Engineer + Data Engineer |
| Micaela | Business Analyst |
| Gaby | Tech Writer + PM |
| Mauro | Support |
