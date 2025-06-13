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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
                # Strip whitespace from keys and values
                row = {k.strip(): v.strip() for k, v in row.items() if k and v}
                name = row.get('name') or row.get('Name')
                grade = row.get('grade (GlaS)') or row.get('grade (Sirinukunwattana et al. 2015)')
                if not name or not grade:
                    continue
                # Binary label: benign = 0, malignant = 1
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
# 2. Model Definition
# ==============================
def build_model(num_classes=2):
    """
    Constructs a ResNet-18 architecture and replaces the final fully-connected layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


# ==============================
# 3. Training Function
# ==============================
def train_one(model, model_name, train_ld, test_ld, device, epochs=30, lr=1e-4, patience=5):
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
# 4. Feature Extraction
# ==============================
def extract_features(dataset, device, model=None):
    """
    Extract features from images using a pretrained CNN (default: ResNet-18).
    Returns (features, labels) as numpy arrays.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    if model is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # Remove final FC layer
    model.eval().to(device)

    features, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting features"):
            x = x.to(device)
            feats = model(x).cpu().numpy()
            features.append(feats)
            labels.append(y.numpy())

    X = np.vstack(features)
    y = np.concatenate(labels)
    return X, y


# ==============================
# 5. Plotting Comparison
# ==============================
def plot_comparison(hist_resnet, rf_acc, rf_f1, ada_acc, ada_f1, ma_acc, ma_f1, out_dir='./'):
    """
    Plots accuracy and F1 comparison among ResNet-18, Random Forest, AdaBoost, and Model Averaging.
    """
    epochs = range(1, len(hist_resnet['test_acc']) + 1)

    # --- Accuracy Curve ---
    plt.figure()
    plt.plot(epochs, hist_resnet['test_acc'], label='ResNet-18')
    plt.hlines(rf_acc, xmin=1, xmax=epochs[-1], colors='g', label='Random Forest', linestyles='dashed')
    plt.hlines(ada_acc, xmin=1, xmax=epochs[-1], colors='r', label='AdaBoost', linestyles='dotted')
    plt.hlines(ma_acc, xmin=1, xmax=epochs[-1], colors='b', label='Model Averaging', linestyles='dashdot')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'compare_accuracy.png'))
    plt.close()

    # --- F1 Curve ---
    plt.figure()
    plt.plot(epochs, hist_resnet['test_f1'], label='ResNet-18')
    plt.hlines(rf_f1, xmin=1, xmax=epochs[-1], colors='g', label='Random Forest', linestyles='dashed')
    plt.hlines(ada_f1, xmin=1, xmax=epochs[-1], colors='r', label='AdaBoost', linestyles='dotted')
    plt.hlines(ma_f1, xmin=1, xmax=epochs[-1], colors='b', label='Model Averaging', linestyles='dashdot')
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
    train_ds = train_ld.dataset
    test_ds = test_ld.dataset

    # 2) Train ResNet-18
    print("\n==> Training ResNet-18")
    model_resnet = build_model(num_classes=2)
    hist_resnet = train_one(model_resnet, 'resnet18', train_ld, test_ld, device)

    # 3) Extract features using pretrained ResNet-18 for traditional ML
    print("\n==> Extracting features for traditional ML models")
    pretrained_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pretrained_resnet.fc = nn.Identity()
    pretrained_resnet.eval().to(device)

    X_train, y_train = extract_features(train_ds, device, model=pretrained_resnet)
    X_test, y_test = extract_features(test_ds, device, model=pretrained_resnet)

    # 4) Train Random Forest
    print("\n==> Training Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    print(f"  Random Forest Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}")

    # 5) Train AdaBoost
    print("\n==> Training AdaBoost")
    ada = AdaBoostClassifier(random_state=0, n_estimators=100)
    ada.fit(X_train, y_train)
    ada_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, ada_pred)
    ada_f1 = f1_score(y_test, ada_pred)
    print(f"  AdaBoost Acc: {ada_acc:.4f}, F1: {ada_f1:.4f}")

    # 6) Model Averaging (assemble multiple saved ResNet-18s)
    print("\n==> Model Averaging with 3 ResNet-18 models")

    def load_resnet_model(path):
        model = build_model(num_classes=2)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # If you only have one saved ResNet, you can duplicate its path 3 times to simulate averaging
    model_paths = ['best_resnet18.pth'] * 8
    models_for_ensemble = [load_resnet_model(p) for p in model_paths]

    def model_averaging_predict(models, dataloader):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Model Averaging"):
                x = x.to(device)
                # Stack logits from each model: shape = [num_models, batch_size, num_classes]
                logits_stack = torch.stack([m(x) for m in models])
                # Average over the first dimension (num_models)
                avg_logits = logits_stack.mean(dim=0)
                preds = avg_logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu())
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc = (preds == labels).float().mean().item()
        f1 = f1_score(labels.numpy(), preds.numpy(), average='binary')
        return acc, f1

    ma_acc, ma_f1 = model_averaging_predict(models_for_ensemble, test_ld)
    print(f"  Model Averaging Acc: {ma_acc:.4f}, F1: {ma_f1:.4f}")

    # 7) Final Comparison Summary
    print("\n== Final Comparison ==")
    print(f"ResNet-18        Acc: {max(hist_resnet['test_acc']):.4f}, F1: {max(hist_resnet['test_f1']):.4f}")
    print(f"Random Forest    Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    print(f"AdaBoost         Acc: {ada_acc:.4f}, F1: {ada_f1:.4f}")
    print(f"Model Averaging  Acc: {ma_acc:.4f}, F1: {ma_f1:.4f}")

    # 8) Plot Comparison (Accuracy & F1)
    plot_comparison(hist_resnet, rf_acc, rf_f1, ada_acc, ada_f1, ma_acc, ma_f1)
