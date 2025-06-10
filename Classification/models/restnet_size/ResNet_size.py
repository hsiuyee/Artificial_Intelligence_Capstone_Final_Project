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
from sklearn.metrics import f1_score

# --- Dataset that splits images into 'train' or 'test' based on filename ---
class CRCClassificationDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, split=None):
        self.img_dir = img_dir
        self.labels = {}

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


def get_data_loaders(img_dir, csv_path, input_size=224, batch_size=32,
                     num_workers=0, seed=42):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CRCClassificationDataset(img_dir, csv_path, transform, split='train')
    test_ds  = CRCClassificationDataset(img_dir, csv_path, transform, split='test')

    torch.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Found {len(train_ds)} train / {len(test_ds)} test samples")
    return train_ld, test_ld


def build_model(model_name='resnet18', num_classes=2):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    for param in model.conv1.parameters():
        param.requires_grad = False

    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def train_one(model, model_name, train_ld, test_ld, device, epochs=30, lr=1e-4, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=False
    )

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_f1': []}
    best_acc, no_improve = 0.0, 0

    model.to(device)
    for ep in range(1, epochs+1):
        print(f"\nEpoch {ep}/{epochs} for {model_name}:")
        # Train
        model.train()
        run_loss, run_corr = 0.0, 0
        for x, y in tqdm(train_ld, desc=' Training', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x.size(0)
            run_corr += (logits.argmax(1) == y).sum().item()

        tr_loss = run_loss / len(train_ld.dataset)
        tr_acc  = run_corr / len(train_ld.dataset)
        print(f"  Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}")

        # Evaluate
        model.eval()
        test_sum, all_pred, all_lab = 0, [], []
        for x, y in tqdm(test_ld, desc=' Testing', leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_sum += criterion(logits, y).item() * x.size(0)
            all_pred.append(logits.argmax(1).cpu())
            all_lab.append(y.cpu())

        te_loss = test_sum / len(test_ld.dataset)
        preds = torch.cat(all_pred).numpy()
        labs  = torch.cat(all_lab).numpy()
        te_acc = (preds == labs).mean()
        te_f1  = f1_score(labs, preds, average='binary')
        print(f"  Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}, F1: {te_f1:.4f}")

        scheduler.step(te_acc)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
        history['test_f1'].append(te_f1)

        if te_acc > best_acc:
            best_acc, no_improve = te_acc, 0
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping...")
                break

    return history


def plot_comparison(histories, metric, out_path):
    plt.figure()
    for name, h in histories.items():
        plt.plot(range(1, len(h[metric])+1), h[metric], label=name)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved comparison for {metric} to {out_path}")


if __name__ == '__main__':
    base    = os.path.dirname(__file__)
    img_dir = os.path.join(base, '../../../Preprocess', 'Warwick_QU_Dataset')
    csv_path= os.path.join(img_dir, 'Grade.csv')

    device = torch.device('mps' if torch.backends.mps.is_available() else \
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_ld, test_ld = get_data_loaders(img_dir, csv_path)

    variants = ['resnet18', 'resnet34', 'resnet50']
    histories = {}
    for v in variants:
        print(f"\n==> Training {v}...")
        histories[v] = train_one(build_model(v), v, train_ld, test_ld, device)

    # Plot all metrics
    for metric in histories[variants[0]].keys():
        plot_comparison(histories, metric, f'compare_{metric}.png')
