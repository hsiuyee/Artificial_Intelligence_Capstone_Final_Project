import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score  # add this import

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
        self.labels = {}

        # Read CSV and build a mapping from basename -> 0/1 label
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

        # Filter filenames: exclude any containing 'anno', then require 'train' or 'test'
        self.img_names = []
        for base in self.labels:
            if 'anno' in base:
                continue
            if split == 'train' and 'train' not in base:
                continue
            if split == 'test'  and 'test'  not in base:
                continue
            path = os.path.join(img_dir, base + '.bmp')
            if os.path.exists(path):
                self.img_names.append(base)

        if not self.img_names:
            raise FileNotFoundError(f"No .bmp files for split='{split}' in {img_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        base = self.img_names[idx]
        path = os.path.join(self.img_dir, base + '.bmp')
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[base]

# --- Build DataLoaders for train & test sets ---
def get_data_loaders(img_dir, csv_path,
                     input_size=224, batch_size=32,
                     num_workers=4, seed=42):
    # Data augmentation & normalization
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
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

# --- Load a pretrained ResNet-18 and replace its head for 2 classes ---
def build_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    # Optionally freeze the very first conv layer
    for param in model.conv1.parameters():
        param.requires_grad = False
    # Replace final fully-connected layer
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def train(model, train_ld, test_ld, device,
          epochs=30, lr=1e-4, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    history = {'train_loss': [], 'test_acc': [], 'test_f1': []}
    best_acc, no_improve = 0.0, 0

    for ep in range(1, epochs+1):
        # ---- Train ----
        model.train()
        running_loss = 0
        for x, y in tqdm(train_ld, desc=f"Epoch {ep}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ld.dataset)

        # ---- Test: compute accuracy + F1 ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(test_ld, desc=f"Epoch {ep}/{epochs} [Test ]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
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
            torch.save(model.state_dict(), 'best_model_pretraining.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    return history, best_acc

# --- Plot metrics (loss, acc, f1) and save ---
def plot_all_metrics(history, out_path='all_metrics.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_acc'],   label='Test Acc')
    plt.plot(epochs, history['test_f1'],    label='Test F1')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved combined metrics plot to {out_path}")

# --- Plot feature importance from final fc layer weights ---
def plot_feature_importance(model, out_path='feature_importance.png', topk=10):
    # Extract weight matrix of final linear layer
    # model.fc is Sequential([Linear, ReLU, Dropout, Linear])
    final_linear = model.fc[-1]
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

    history, best_acc = train(model, train_ld, test_ld, device)
    print(f"Best Test Acc: {best_acc:.4f}")

    # Save metric plots and feature importance
    plot_all_metrics(history, out_path='metrics.png')
    plot_feature_importance(model, out_path='feature_importance.png', topk=10)
