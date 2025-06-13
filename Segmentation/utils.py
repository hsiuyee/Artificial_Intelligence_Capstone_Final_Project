import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
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
class StandardGlaSDataset(Dataset):
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


class FewShotSegContrastiveDataset(Dataset):
    def __init__(self, dataset_root, transform=None, mode="train", k=10):
        """
        dataset_root: 根目錄，例如 'dataset'
        mode: 'train' or 'val'
        k: few-shot 數量 (僅在 train 模式下使用)
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.mode = mode
        self.k = k

        self.image_dir = os.path.join(self.dataset_root, "images", self.mode)
        self.mask_dir = os.path.join(self.dataset_root, "masks", self.mode)

        self.images = os.listdir(self.image_dir)

        if self.mode == "train":
            self._generate_pairs()
        else:
            self.test_images = self.images

    def _generate_pairs(self):
        positive = []
        negative = []

        for name in self.images:
            mask_path = os.path.join(self.mask_dir, name)
            mask = np.array(Image.open(mask_path).convert("L"))
            if np.sum(mask) > 0:
                positive.append(name)
            else:
                negative.append(name)

        random.shuffle(positive)
        random.shuffle(negative)

        k_pos = positive[:self.k // 2]
        k_neg = negative[:self.k - (self.k // 2)]

        print(f"Total positive images: {len(positive)}")
        print(f"Total negative images: {len(negative)}")
        print(f"Selected {len(k_pos)} positive and {len(k_neg)} negative for k-shot")

        self.pos_pairs = [(a, b, 0) for i, a in enumerate(k_pos) for j, b in enumerate(k_pos) if i < j]
        self.neg_pairs = [(a, b, 1) for a in k_pos for b in k_neg]
        self.all_pairs = self.pos_pairs + self.neg_pairs

    def __len__(self):
        if self.mode == "train":
            return len(self.all_pairs)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        if self.mode == "train":

            img1_name, img2_name, label = self.all_pairs[idx]

            img1_path = os.path.join(self.image_dir, img1_name)
            img2_path = os.path.join(self.image_dir, img2_name)
            mask1_path = os.path.join(self.mask_dir, img1_name)
            mask2_path = os.path.join(self.mask_dir, img2_name)

            img1 = np.array(Image.open(img1_path).convert("RGB"))
            img2 = np.array(Image.open(img2_path).convert("RGB"))
            mask1 = np.array(Image.open(mask1_path).convert("L"))
            mask2 = np.array(Image.open(mask2_path).convert("L"))

            mask1 = (mask1 > 0).astype(np.float32)
            mask2 = (mask2 > 0).astype(np.float32)

            if self.transform:
                aug1 = self.transform(image=img1, mask=mask1)
                img1, mask1 = aug1['image'], aug1['mask'].unsqueeze(0)

                aug2 = self.transform(image=img2, mask=mask2)
                img2, mask2 = aug2['image'], aug2['mask'].unsqueeze(0)

            return img1, mask1, img2, mask2, label

        else:  # test 模式
            img_name = self.test_images[idx]
            img = np.array(Image.open(os.path.join(self.image_dir, img_name)).convert("RGB"))
            mask = np.array(Image.open(os.path.join(self.mask_dir, img_name)).convert("L"))
            mask = (mask > 0).astype(np.float32)

            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].unsqueeze(0)

            return img, mask


def get_dataloader():

    train_dataset = StandardGlaSDataset("dataset/images/train", "dataset/masks/train", transform=get_transform(train=True))
    val_dataset = StandardGlaSDataset("dataset/images/val", "dataset/masks/val", transform=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    return train_loader, val_loader


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
