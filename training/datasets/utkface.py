"""
UTKFace Dataset Loader.
Format: [age]_[gender]_[race]_[datetime].jpg
Gender: 0 (Male), 1 (Female)
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

class UTKFaceDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if not self.data_dir.exists():
            return
            
        for img_path in self.data_dir.glob("*.jpg"):
            parts = img_path.stem.split("_")
            if len(parts) >= 3:
                try:
                    age = int(parts[0])
                    gender = int(parts[1])  # 0: male, 1: female
                    if 0 <= gender <= 1:
                        self.image_paths.append(img_path)
                        self.labels.append((age, gender))
                except ValueError:
                    continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        age, gender = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, float(age), gender
