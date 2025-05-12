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

# --- Dataset 不变，只是保留 train/test 分割逻辑 ---
class CRCClassificationDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, split=None):
        self.img_dir = img_dir
        self.labels = {}
        with open(csv_path, newline='') as f:
            sample = f.read(1024); f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters='\t,')
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                row = {k.strip(): v.strip() for k,v in row.items() if k and v}
                name = row.get('name') or row.get('Name')
                gla  = row.get('grade (GlaS)') or row.get('grade (Sirinukunwattana et al. 2015)')
                if not name or not gla: continue
                lbl = 0 if gla.lower()=='benign' else 1
                base = os.path.splitext(name)[0]
                self.labels[base] = lbl

        self.img_names = []
        for base in self.labels:
            if 'anno' in base: continue
            if split=='train' and 'train' not in base: continue
            if split=='test'  and 'test'  not in base: continue
            path = os.path.join(img_dir, base + '.bmp')
            if os.path.exists(path): self.img_names.append(base)
        if not self.img_names:
            raise FileNotFoundError(f"No .bmp files for split={split}")

        self.transform = transform

    def __len__(self): return len(self.img_names)
    def __getitem__(self, idx):
        base = self.img_names[idx]
        img = Image.open(os.path.join(self.img_dir, base + '.bmp')).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[base]

# --- DataLoader ---
def get_data_loaders(img_dir, csv_path,
                     input_size=224, batch_size=32,
                     num_workers=4, seed=42):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    train_ds = CRCClassificationDataset(img_dir, csv_path, transform, split='train')
    test_ds  = CRCClassificationDataset(img_dir, csv_path, transform, split='test')

    torch.manual_seed(seed)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Found {len(train_ds)} train / {len(test_ds)} test samples")
    return train_ld, test_ld

# --- Model: 用预训练 ResNet18 ---
def build_model(num_classes=2):
    model = models.resnet18(pretrained=True)
    # 冻结前几层（可选），这样微调速度更快：
    for param in model.conv1.parameters():
        param.requires_grad = False
    # 换掉最后全连接
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

# --- Training & Evaluation ---
def train(model, train_ld, test_ld, device,
          epochs=30, lr=1e-4, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    history = {'train_loss': [], 'test_acc': []}
    best_acc, no_improve = 0.0, 0

    for ep in range(1, epochs+1):
        # --- Train ---
        model.train()
        running_loss = 0
        for x,y in tqdm(train_ld, desc=f"Epoch {ep}/{epochs} [Train]"):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ld.dataset)

        # --- Test ---
        model.eval()
        correct = total = 0
        for x,y in tqdm(test_ld, desc=f"Epoch {ep}/{epochs} [Test ]"):
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                preds = model(x).argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
        test_acc = correct / total

        # 打印当前 lr
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}  "
              f"Train Loss: {train_loss:.4f}  Test Acc: {test_acc:.4f}")

        scheduler.step(test_acc)
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)

        # Early stop
        if test_acc > best_acc:
            best_acc, no_improve = test_acc, 0
            torch.save(model.state_dict(), 'best_model_pretraining.pth')
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
    plt.plot(epochs, history['test_acc'],    label='Test Acc')
    plt.xlabel('Epoch'); plt.legend(); plt.show()

# --- Main ---
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
    history, best = train(model, train_ld, test_ld, device)
    print(f"Best Test Acc: {best:.4f}")
    plot_metrics(history)
