# Brain Tumor Classification

ViT-B/16 trained on brain tumor MRI scans (glioma, meningioma, pituitary, no tumor).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchmetrics tqdm matplotlib pillow
```

## Dataset

Place images in `dataset/Training/<class>/` and `dataset/Testing/<class>/`.

## Train

```bash
python main/main_model.py
```

Outputs `tumor_classifier.pth` (PyTorch weights), `tumor_classifier.onnx` (ONNX export), and `outputs/training_curves.png` + `outputs/predictions.png` (visualizations).

## Config

Edit `main/main_model.py` to adjust epochs, batch size, learning rate, etc.
