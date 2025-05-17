import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
import random


def get_transform(train=True):
    if train:
        return A.Compose([
            A.Resize(256, 256),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    smooth = 1e-6

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

# Dataset
class GlaSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # (1, H, W)

        return image, mask


def build_dataset(source_dir, output_dir, split_ratio = 0.8):
    # === Create output folders ===
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks", split), exist_ok=True)

    # === Get all image/mask pairs ===
    files = os.listdir(source_dir)
    base_names = sorted(set(f.replace("_anno", "").replace(".bmp", "") for f in files))

    # === Shuffle and split ===
    random.shuffle(base_names)
    split_idx = int(len(base_names) * split_ratio)
    train_bases = base_names[:split_idx]
    val_bases = base_names[split_idx:]

    def copy_files(base_list, split):
        for base in base_list:
            image_name = f"{base}.bmp"
            mask_name = f"{base}_anno.bmp"

            src_image = os.path.join(source_dir, image_name)
            src_mask = os.path.join(source_dir, mask_name)

            dst_image = os.path.join(output_dir, "images", split, image_name)
            dst_mask = os.path.join(output_dir, "masks", split, image_name)  # rename mask to match image name

            if os.path.exists(src_image) and os.path.exists(src_mask):
                shutil.copyfile(src_image, dst_image)
                shutil.copyfile(src_mask, dst_mask)
            else:
                print(f"[WARN] Missing pair for {base}")

    # === Copy files ===
    copy_files(train_bases, "train")
    copy_files(val_bases, "val")

    print(f"✅ Done. {len(train_bases)} training samples, {len(val_bases)} validation samples.")


if __name__ == "__main__":
    build_dataset(
        source_dir=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../Preprocess/Warwick_QU_Dataset")
        ),
        output_dir="dataset",
        split_ratio=0.8
    )
