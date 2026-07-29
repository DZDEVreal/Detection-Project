import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, transform=test_transform):
        self.image_dir = image_dir
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(image_dir))
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(image_dir, class_name)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(class_dir, file))
                        self.labels.append(self.class_to_idx[class_name])

        self.transform = transform

    @property
    def class_weights(self) -> torch.Tensor:
        labels = torch.tensor(self.labels)
        counts = torch.bincount(labels)
        total = len(labels)
        weights = total / (len(counts) * counts.float())
        return weights

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
