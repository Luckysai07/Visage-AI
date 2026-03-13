"""
FER2013 Dataset Loader.
Expected directory structure:
data/raw/FER2013/train/<emotion_class>/img.jpg
data/raw/FER2013/test/<emotion_class>/img.jpg
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

from app.core.config import settings

class FER2013Dataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = settings.EMOTION_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        if not self.data_dir.exists():
            return
            
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
