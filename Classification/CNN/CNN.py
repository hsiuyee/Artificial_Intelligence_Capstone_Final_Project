import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
# models no longer used, using custom CRCNet
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Custom Classification Dataset ---
class CRCClassificationDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.labels = {}
        with open(csv_path, newline='') as f:
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters='\t,')
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items() if k and v}
                name = row.get('name') or row.get('Name')
                gla_label = row.get('grade (GlaS)') or row.get('grade (Sirinukunwattana et al. 2015)')
                if not name or not gla_label:
                    continue
                label = 0 if gla_label.lower() == 'benign' else 1
                name_base = os.path.splitext(name)[0]
                self.labels[name_base] = label
        self.img_names = [nm for nm in self.labels if os.path.exists(os.path.join(img_dir, nm + '.bmp'))]
        if not self.img_names:
            raise FileNotFoundError(f"No matching .bmp files found in {img_dir} for labels in CSV.")
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        path = os.path.join(self.img_dir, name + '.bmp')
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[name]
        return image, label

# --- Data Loaders ---
def get_data_loaders(img_dir, csv_path, input_size=224, batch_size=32, split_ratio=0.8, seed=42):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    full_ds = CRCClassificationDataset(img_dir, csv_path, transform)
    total = len(full_ds)
    train_len = int(total * split_ratio)
    val_len = total - train_len
    torch.manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Dataset: total={total}, train={train_len}, val={val_len}")
    return train_ld, val_ld, train_len, val_len

# --- Model & Training ---
# Define custom CNN as per the paper "Deep Convolutional Networks"
class CRCNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 5x5 conv -> ReLU -> maxpool
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# build_model now returns CRCNet

def build_model(num_classes=2):
    return CRCNet(num_classes)

# Training function follows
def train(model, train_ld, val_ld, device, epochs=30, lr=1e-4, momentum=0.9, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0
    no_improve = 0

    for ep in range(1, epochs+1):
        # training
        model.train()
        running_loss = 0
        for x, y in tqdm(train_ld, desc=f"Epoch {ep}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ld.dataset)

        # validation
        model.eval()
        correct = total = 0
        for x, y in tqdm(val_ld, desc=f"Epoch {ep}/{epochs} [Val  ]"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        val_acc = correct / total

        print(f"Epoch {ep}/{epochs}  Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}")
        scheduler.step(val_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
    return history, best_acc


def plot_metrics(history):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_acc'],    label='Val Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# --- Main ---
if __name__ == '__main__':
    base = os.path.dirname(__file__)
    img_dir = os.path.join(base, '../../Preprocess', 'Warwick_QU_Dataset')
    csv_path = os.path.join(img_dir, 'Grade.csv')
    
    # select device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    train_ld, val_ld, tr_len, val_len = get_data_loaders(img_dir, csv_path)
    model = build_model().to(device)
    history, best = train(model, train_ld, val_ld, device)
    print(f"Best Val Acc: {best:.4f}")
    plot_metrics(history)

