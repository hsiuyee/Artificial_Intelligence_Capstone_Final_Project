import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score  # add this import

matplotlib.use('Agg')

# --- Dataset that splits images into 'train' or 'test' based on filename ---
class CRCClassificationDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, split=None):
        """
        img_dir  : directory containing .bmp images
        csv_path : path to the labels CSV
        transform: torchvision transforms to apply
        split    : 'train' or 'test' (selects filenames containing that substring)
        """
        self.img_dir = img_dir
        self.type = split
        self.labels = {}
        self.pair = ('train_2', 'train_1')
        self.transform = transform
        self.randomAugment = transforms.RandAugment(2, 11)
        if self.type == 'train':
            return

        # Read CSV and build a mapping from basename -> 0/1 label, only for test_ds
        with open(csv_path, newline='') as f:
            sample = f.read(1024); f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters='\t,')
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items() if k and v}
                name = row.get('name') or row.get('Name')
                grade = row.get('grade (GlaS)') or row.get('grade (Sirinukunwattana et al. 2015)')
                if not name or not grade:
                    continue
                label = 0 if grade.lower() == 'benign' else 1
                base = os.path.splitext(name)[0]
                self.labels[base] = label


        # Filter filenames: exclude any containing 'anno', then require 'train' or 'test', only for test_ds
        self.img_names = []
        for base in self.labels:
            if 'anno' in base:
                continue
            if split == 'test'  and (base == 'train_1' or base == 'train_2'):
                continue
            path = os.path.join(img_dir, base + '.bmp')
            if os.path.exists(path):
                self.img_names.append(base)

        if not self.img_names:
            raise FileNotFoundError(f"No .bmp files for split='{split}' in {img_dir}")

    def __len__(self):
        if self.type == 'train':
            return 800
        else:
            return len(self.img_names) * 2

    def __getitem__(self, idx):
        lab = random.randint(0, 1)
        if self.type == 'train':
            if lab == 0:
                idx = idx % 2
                base = self.pair[idx]
                path = os.path.join(self.img_dir, base + '.bmp')
                img1 = Image.open(path).convert('RGB')
                img2 = img1.copy()
            else:
                idx = idx % 2
                base = self.pair[idx]
                path = os.path.join(self.img_dir, base + '.bmp')
                img1 = Image.open(path).convert('RGB')
                idx = 1 - idx
                base = self.pair[idx]
                path = os.path.join(self.img_dir, base + '.bmp')
                img2 = Image.open(path).convert('RGB')
            if self.transform:
                img1 = self.randomAugment(img1)
                img1 = self.transform(img1)
                img2 = self.randomAugment(img2)
                img2 = self.transform(img2)
        else:
            base = self.pair[idx % 2]
            path = os.path.join(self.img_dir, base + '.bmp')
            img2 = Image.open(path).convert('RGB')
            base = self.img_names[idx // 2]
            path = os.path.join(self.img_dir, base + '.bmp')
            img1 = Image.open(path).convert('RGB')
            lab = 0 if self.labels[base] == idx % 2 else 1
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        return img1, img2, lab

# --- Build DataLoaders for train & test sets ---
def get_data_loaders(img_dir, csv_path,
                     input_size=224, batch_size=32,
                     num_workers=0, seed=42):
    # Data augmentation & normalization
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CRCClassificationDataset(img_dir, csv_path, transform, split='train')
    test_ds  = CRCClassificationDataset(img_dir, csv_path, transform, split='test')

    torch.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Found {len(train_ds)} train / {len(test_ds)} test samples")
    return train_ld, test_ld

# --- The function of contrastive loss ---
def criterion(x1, x2, label, margin: float = 0.4):
    """
    Computes Contrastive Loss
    """
    dist = torch.nn.functional.pairwise_distance(x1, x2)
    print(label, dist)
    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

# --- Load a pretrained AlexNet and replace its classifier ---
def build_model(embedding_len = 256):

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # Optionally freeze the very first conv layer
    for param in model.features[0].parameters():
        param.requires_grad = False
    """
    # Replace final classifier(fc) layer
    model.classifier = nn.Sequential(
        nn.Linear(256*6*6, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(2048, embedding_len)
    )"""
    return model

def train(model, train_ld, test_ld, device,
          epochs=30, lr=1e-4, patience=7, margin=0.2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience
    )


    history = {'train_loss': [], 'test_acc': [], 'test_f1': []}
    best_acc, no_improve = 0.0, 0

    for ep in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running_loss = 0
        for x1, x2, y in tqdm(train_ld, desc=f"Epoch {ep}/{epochs} [Train]"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x1), model(x2), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x1.size(0)
        train_loss = running_loss / len(train_ld.dataset)

        # ---- Test: compute accuracy + F1 ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x1, x2, y in tqdm(test_ld, desc=f"Epoch {ep}/{epochs} [Test ]"):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                dist = torch.nn.functional.pairwise_distance(model(x1), model(x2), keepdim=False)
                preds = (dist >= margin)
                print(dist, preds, y)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        test_acc = (all_preds == all_labels).mean()
        test_f1  = f1_score(all_labels, all_preds, average='binary')

        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.2e}  "
              f"Train Loss: {train_loss:.4f}  "
              f"Test Acc: {test_acc:.4f}  "
              f"Test F1: {test_f1:.4f}")

        # Step scheduler and record history
        scheduler.step(test_acc)
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)

        # Save best model
        if test_acc > best_acc:
            best_acc, no_improve = test_acc, 0
            torch.save(model.state_dict(), './Classification/One-shot/best_model_pretraining.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    return history, best_acc

# --- Plot metrics (loss, acc, f1) and save ---
def plot_all_metrics(history, out_path='./'):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path+"loss.png")
    plt.close()
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history['test_acc'],   label='Test Acc')
    plt.plot(epochs, history['test_f1'],    label='Test F1')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path+"acc_F1.png")
    plt.close()
    print(f"Saved combined metrics plot to {out_path}")

# --- Plot feature importance from final fc layer weights ---
def plot_feature_importance(model, out_path='feature_importance.png', topk=10):
    # Extract weight matrix of final linear layer
    # model.fc is Sequential([Linear, ReLU, Dropout, Linear])
    final_linear = model.classifier
    weights = final_linear.weight.data.cpu().abs()  # shape: (num_classes, feat_dim)
    # Sum across classes to get per-feature importance
    importance = weights.sum(dim=0)  # shape: (feat_dim,)
    # Get top-k feature indices
    topk_vals, topk_idx = importance.topk(topk)
    plt.figure(figsize=(8,4))
    plt.bar(range(topk), topk_vals.numpy(), tick_label=topk_idx.numpy())
    plt.xlabel('Feature Index')
    plt.ylabel('Sum |Weight|')
    plt.title(f'Top {topk} Feature Importances')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature importance plot to {out_path}")

# --- Main entrypoint ---
if __name__ == '__main__':
    base    = os.path.dirname(__file__)
    img_dir = os.path.join(base, '../../Preprocess', 'Warwick_QU_Dataset')
    csv_path= os.path.join(img_dir, 'Grade.csv')

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("Using device:", device)

    train_ld, test_ld = get_data_loaders(img_dir, csv_path)
    model = build_model().to(device)

    try:
        history, best_acc = train(model, train_ld, test_ld, device)
    except KeyboardInterrupt:
        print("KeyboardInturrupt")
        if device == 'cuda':
            torch.cuda.empty_cache()
        os._exit(1)
    print(f"Best Test Acc: {best_acc:.4f}")

    # Save metric plots and feature importance
    plot_all_metrics(history, out_path='./Classification/One-shot/')
    # plot_feature_importance(model, out_path='feature_importance.png', topk=10)
