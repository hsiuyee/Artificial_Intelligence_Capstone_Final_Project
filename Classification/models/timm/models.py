import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm  # PyTorch Image Models
from sklearn.metrics import f1_score
import numpy as np

# ==============================
# 1. Dataset Definition
# ==============================
class CRCClassificationDataset(Dataset):
    """
    Reads image filenames and grades from a CSV, then splits into 'train' or 'test'
    based on whether the filename contains 'train' or 'test'. Expects CSV columns
    "name" and "grade (GlaS)".
    """
    def __init__(self, img_dir, csv_path, transform=None, split=None):
        self.img_dir = img_dir
        self.labels = {}

        # Read CSV (handle TSV or CSV with mixed delimiters)
        with open(csv_path, newline='') as f:
            sample = f.read(1024)
            f.seek(0)
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

        # Filter filenames by split
        self.img_names = []
        for base in self.labels:
            if 'anno' in base:
                continue
            if split == 'train' and 'train' not in base:
                continue
            if split == 'test' and 'test' not in base:
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


def get_data_loaders(img_dir, csv_path, input_size=224, batch_size=32,
                     num_workers=2, seed=42):
    """
    Creates and returns train and test DataLoaders for CRCClassificationDataset.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CRCClassificationDataset(img_dir, csv_path, transform, split='train')
    test_ds = CRCClassificationDataset(img_dir, csv_path, transform, split='test')

    torch.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Found {len(train_ds)} train / {len(test_ds)} test samples")
    return train_ld, test_ld


# ==============================
# 2. Model Factory
# ==============================
def build_model(model_name, num_classes=2, use_pretrained=True):
    """
    Creates a model given its name from timm. For models without pretrained weights,
    use pretrained=False automatically.
    """
    # nfnet_f0 does not have pretrained weights; use pretrained=False
    if model_name.startswith('nfnet_'):
        pretrained = False
    else:
        pretrained = use_pretrained

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


# ==============================
# 3. Training Function
# ==============================
def train_one(model, model_name, train_ld, test_ld, device, epochs=2, lr=1e-4, patience=5):
    """
    Trains the model for a specified number of epochs using Adam optimizer and
    ReduceLROnPlateau scheduler (monitoring test accuracy). Implements early stopping.
    Returns a dictionary containing training/validation history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=3, verbose=False
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [], 'test_f1': []
    }
    best_acc, no_improve = 0.0, 0

    model.to(device)
    for ep in range(1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs} for {model_name}:")

        # --------------------
        # Training Phase
        # --------------------
        model.train()
        run_loss, run_corr = 0.0, 0
        for x, y in tqdm(train_ld, desc='Training', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            run_corr += (logits.argmax(1) == y).sum().item()

        tr_loss = run_loss / len(train_ld.dataset)
        tr_acc = run_corr / len(train_ld.dataset)
        print(f"  Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")

        # --------------------
        # Evaluation Phase
        # --------------------
        model.eval()
        test_sum, all_pred, all_lab = 0, [], []
        with torch.no_grad():
            for x, y in tqdm(test_ld, desc='Testing', leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_sum += criterion(logits, y).item() * x.size(0)
                all_pred.append(logits.argmax(1).cpu())
                all_lab.append(y.cpu())

        te_loss = test_sum / len(test_ld.dataset)
        preds = torch.cat(all_pred).numpy()
        labs = torch.cat(all_lab).numpy()
        te_acc = (preds == labs).mean()
        te_f1 = f1_score(labs, preds, average='binary')
        print(f"  Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}, F1: {te_f1:.4f}")

        scheduler.step(te_acc)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        history['test_f1'].append(te_f1)

        # Early stopping
        if te_acc > best_acc:
            best_acc, no_improve = te_acc, 0
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping...")
                break

    return history


# ==============================
# 4. Plotting Comparison
# ==============================
def plot_comparison(histories, out_dir='./'):
    """
    Given a dict mapping model_name -> history dict, plots:
      - test_acc vs epoch for all models
      - test_f1  vs epoch for all models
    Saves as 'compare_accuracy.png' and 'compare_f1.png' in out_dir.
    """
    epochs = range(1, len(next(iter(histories.values()))['test_acc']) + 1)

    # --- Accuracy Curve ---
    plt.figure()
    for name, h in histories.items():
        plt.plot(epochs, h['test_acc'], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'compare_accuracy.png'))
    plt.close()

    # --- F1 Curve ---
    plt.figure()
    for name, h in histories.items():
        plt.plot(epochs, h['test_f1'], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Model F1 Score Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'compare_f1.png'))
    plt.close()

    print(f"Plots saved to {out_dir}")


# ==============================
# 6. Main Script
# ==============================
if __name__ == '__main__':
    base = os.path.dirname(__file__)
    img_dir = os.path.join(base, '../../../Preprocess', 'Warwick_QU_Dataset')
    csv_path = os.path.join(img_dir, 'Grade.csv')

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("Using device:", device)

    # 1) Load data
    train_ld, test_ld = get_data_loaders(img_dir, csv_path, num_workers=0)

    # 2) Define model names to compare
    model_list = [
        'resnet18',
        'efficientnet_b0',
        'nfnet_f0',                    # no pretrained weights
        'vit_base_patch16_224',
        'mobilenetv3_large_100',       # 近似 MobileNetV4
        'dpn68',                       # DPN
        'cspresnet50',                 # CSPNet
        'swin_tiny_patch4_window7_224' # Swin Transformer
    ]

    histories = {}

    # 3) Loop over each architecture
    for model_name in model_list:
        print(f"\n==> Training {model_name} ==")
        model = build_model(model_name, num_classes=2, use_pretrained=True)
        history = train_one(model, model_name, train_ld, test_ld, device)
        histories[model_name] = history

    # 4) Final accuracy & F1 printout
    print("\n== Final Comparison ==")
    for name, h in histories.items():
        best_epoch = np.argmax(h['test_acc']) + 1
        best_acc = h['test_acc'][best_epoch - 1]
        best_f1 = h['test_f1'][best_epoch - 1]
        print(f"{name:<25} Best Epoch {best_epoch:>2}  Acc: {best_acc:.4f}, F1: {best_f1:.4f}")

    # 5) Plot comparison charts
    plot_comparison(histories, out_dir='.')
