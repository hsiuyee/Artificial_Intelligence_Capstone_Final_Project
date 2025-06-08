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


class FewShotGlaSDataset(Dataset):
    def __init__(self, split='train', transform=None, shot=1):
        self.img_dir = os.path.join("dataset", 'images', split)
        self.mask_dir = os.path.join("dataset", 'masks', split)
        self.transform = transform
        self.shot = shot

        self.img_list = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        query_name = self.img_list[idx]
        query_img = Image.open(os.path.join(self.img_dir, query_name)).convert('RGB')
        query_mask = Image.open(os.path.join(self.mask_dir, query_name)).convert('L')

        support_indices = random.sample([i for i in range(len(self)) if i != idx], self.shot)
        support_imgs, support_masks = [], []

        # === Apply transform on QUERY ===
        if self.transform:
            augmented = self.transform(image=np.array(query_img), mask=np.array(query_mask))
            query_img = augmented["image"]
            query_mask = augmented["mask"].unsqueeze(0)
        else:
            query_img = transforms.ToTensor()(query_img)
            query_mask = transforms.ToTensor()(query_mask)
            query_mask = (query_mask > 0.5).float()

        # === Apply transform on SUPPORT(s) ===
        for i in support_indices:
            s_img = Image.open(os.path.join(self.img_dir, self.img_list[i])).convert('RGB')
            s_mask = Image.open(os.path.join(self.mask_dir, self.img_list[i])).convert('L')

            if self.transform:
                augmented = self.transform(image=np.array(s_img), mask=np.array(s_mask))
                s_img = augmented["image"]
                s_mask = augmented["mask"].unsqueeze(0)
            else:
                s_img = transforms.ToTensor()(s_img)
                s_mask = transforms.ToTensor()(s_mask)
                s_mask = (s_mask > 0.5).float()

            support_imgs.append(s_img)
            support_masks.append(s_mask)

        support_imgs = torch.stack(support_imgs)
        support_masks = torch.stack(support_masks)

        return support_imgs, support_masks, query_img, query_mask



def get_dataloader(shot):
    def custom_collate_fn(batch):
        return tuple(zip(*batch))

    if shot == -1:
        train_dataset = StandardGlaSDataset("dataset/images/train", "dataset/masks/train", transform=get_transform(train=True))
        val_dataset = StandardGlaSDataset("dataset/images/val", "dataset/masks/val", transform=get_transform(train=False))
    else:
        train_dataset = FewShotGlaSDataset(split="train", transform=get_transform(train=True), shot=shot)
        val_dataset   = FewShotGlaSDataset(split="val", transform=get_transform(train=False), shot=shot)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

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
