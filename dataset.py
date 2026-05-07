# This part proves: i created a custom dataset class and handled images + masks loading
# dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Get sorted list of images (important for consistency)
        self.images = sorted([
            img for img in os.listdir(image_dir)
            if img.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Paths
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read files
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Error handling
        if image is None:
            raise ValueError(f"Image not found: {img_path}")
        if mask is None:
            raise ValueError(f"Mask not found: {mask_path}")

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize (IMPORTANT: correct interpolation for mask)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Normalize image [0,255] → [0,1]
        image = image / 255.0

        # Convert to tensor and change shape (H,W,C → C,H,W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Convert mask to binary tensor (0 or 1)
        mask = (mask > 0).astype("float32")
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask