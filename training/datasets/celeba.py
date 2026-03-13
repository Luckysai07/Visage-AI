"""
CelebA Dataset Loader.
Requires:
- img_align_celeba/ folder with images
- list_attr_celeba.txt with 40 binary attributes
"""

import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple

from app.core.config import settings

class CelebADataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "img_align_celeba"
        self.attr_file = self.data_dir / "list_attr_celeba.txt"
        self.eval_file = self.data_dir / "list_eval_partition.txt"
        self.transform = transform
        
        self.valid = False
        if not (self.img_dir.exists() and self.attr_file.exists() and self.eval_file.exists()):
            return
            
        # 0: train, 1: val, 2: test
        split_map = {"train": 0, "val": 1, "test": 2}
        split_idx = split_map.get(split, 0)
        
        # Load partitions
        eval_df = pd.read_csv(self.eval_file, sep=r'\s+', header=None, names=["image_id", "partition"])
        split_images = set(eval_df[eval_df["partition"] == split_idx]["image_id"])
        
        # Load attributes
        # CelebA attr file has: Row 1 = count, Row 2 = header, Row 3+ = data
        attr_df = pd.read_csv(self.attr_file, sep=r'\s+', header=1)
        
        # Filter by partition
        self.data = attr_df[attr_df.index.isin(split_images)].copy()
        
        # CelebA uses -1 for False, 1 for True. Convert to 0/1.
        for col in self.data.columns:
            self.data[col] = (self.data[col] == 1).astype(int)
            
        self.image_names = self.data.index.tolist()
        # Ensure column order matches settings.CELEBA_ATTRIBUTES
        self.labels = self.data[settings.CELEBA_ATTRIBUTES].values
        self.valid = True

    def __len__(self):
        return len(self.image_names) if self.valid else 0

    def __getitem__(self, idx: int) -> Tuple:
        img_name = self.image_names[idx]
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert("RGB")
        
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels
