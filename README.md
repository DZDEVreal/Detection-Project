# Brain Tumor Classification

Swin-T tiny trained on brain tumor MRI scans (glioma, meningioma, pituitary, no tumor). Achieves ~80% test accuracy.

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

Outputs `tumor_classifier.pth` (weights), `tumor_classifier.onnx` (ONNX), and `outputs/training_curves.png` + `outputs/predictions.png`.

## Config

Edit `main/main_model.py` to adjust epochs, batch size, learning rate, etc.
