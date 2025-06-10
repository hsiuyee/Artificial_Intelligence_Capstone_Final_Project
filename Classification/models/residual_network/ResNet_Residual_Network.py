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
    # transform = transforms.Compose([
    #     transforms.Resize((input_size, input_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225]),
    # ])

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
# 2. Custom Residual Blocks
# ==============================

# 2.1. Alpha-Weighted Residual Block (BasicBlockAlpha)
class BasicBlockAlpha(nn.Module):
    """
    BasicBlock variant where y = F(x) + x is replaced by y = F(x) + alpha * x.
    Accepts all arguments that torchvision's ResNet might pass.
    """
    expansion = 1  # For ResNet-18 / ResNet-34

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, alpha=1.0):
        """
        inplanes:   Number of input channels
        planes:     Number of output channels for the residual branch
        stride:     Stride for the first convolution
        downsample: If not None, a nn.Sequential of 1x1 conv + BN to match dimensions
        groups:     Groups for convolution (unused in BasicBlock, but kept for compatibility)
        base_width: Base width (unused in BasicBlock, but kept for compatibility)
        dilation:   Dilation for convolution (unused in BasicBlock, but kept for compatibility)
        norm_layer: Normalization layer (defaults to BatchNorm2d)
        alpha:      Weight for the identity shortcut
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.alpha = alpha

        # First 3x3 convolution + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        # Second 3x3 convolution + BatchNorm
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        # Optional downsample for matching dimensions
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Weighted identity shortcut: y = F(x) + alpha * x
        out = out + self.alpha * identity
        out = self.relu(out)
        return out


# 2.2. Pre-Activation Residual Block (PreActBasicBlock)
class PreActBasicBlock(nn.Module):
    """
    Pre-activation BasicBlock (ResNet v2 style). Implements:
    y = x + W2 * σ(BN(W1 * σ(BN(x)))), without an extra ReLU after addition.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        inplanes:   Number of input channels
        planes:     Number of output channels for the residual branch
        stride:     Stride for the first convolution
        downsample: If not None, a nn.Sequential of 1x1 conv + BN to match dimensions
        groups:     Groups for convolution (unused in BasicBlock, but kept for compatibility)
        base_width: Base width (unused in BasicBlock, but kept for compatibility)
        dilation:   Dilation for convolution (unused in BasicBlock, but kept for compatibility)
        norm_layer: Normalization layer (defaults to BatchNorm2d)
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # Pre-activation: BN -> ReLU before first convolution
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        # Pre-activation before second convolution
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)

        identity = x
        if self.downsample is not None:
            # Apply downsample to the activated x
            identity = self.downsample(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # Add identity without extra ReLU
        out += identity
        return out


# 2.3. Gated Residual Block (BasicBlockGated)
class BasicBlockGated(nn.Module):
    """
    Gated Residual Block implementing:
      y = G(x) ⊙ F(x) + (1 - G(x)) ⊙ x,
    where G(x) = sigmoid(BN(Conv1x1(x))).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        inplanes:   Number of input channels
        planes:     Number of output channels for the residual branch
        stride:     Stride for the first convolution in residual and gate branches
        downsample: If not None, a nn.Sequential of 1x1 conv + BN to match dimensions
        groups:     Groups for convolution (unused in BasicBlock, but kept for compatibility)
        base_width: Base width (unused in BasicBlock, but kept for compatibility)
        dilation:   Dilation for convolution (unused in BasicBlock, but kept for compatibility)
        norm_layer: Normalization layer (defaults to BatchNorm2d)
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # Residual branch F(x): two 3x3 convolutions
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        # Gate branch G(x): 1x1 convolution + BN + Sigmoid
        self.gate_conv = nn.Conv2d(inplanes, planes, kernel_size=1,
                                   stride=stride, bias=False)
        self.gate_bn = norm_layer(planes)
        self.gate_sigmoid = nn.Sigmoid()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # Compute G(x)
        gate = self.gate_conv(x)
        gate = self.gate_bn(gate)
        gate = self.gate_sigmoid(gate)  # Shape: [B, planes, H, W]

        # Residual branch F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Identity branch
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # Gated combination: y = G(x)*F(x) + (1 - G(x))*identity
        y = gate * out + (1.0 - gate) * identity
        y = self.relu(y)
        return y


# ==============================
# 3. Helper Functions to Build Modified ResNet-18
# ==============================

def build_resnet18_alpha(alpha=1.0, num_classes=2):
    """
    Constructs a ResNet-18 architecture using BasicBlockAlpha (y = F(x) + alpha*x).
    """
    layers = [2, 2, 2, 2]  # Number of blocks in each of the four layers

    class ResNet18Alpha(models.resnet.ResNet):
        def __init__(self):
            # Pass BasicBlockAlpha as the block type
            super().__init__(block=BasicBlockAlpha, layers=layers, num_classes=num_classes)
            # Replace the final fully connected layer
            in_feat = self.fc.in_features
            self.fc = nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            # Assign alpha to each BasicBlockAlpha instance
            for m in self.modules():
                if isinstance(m, BasicBlockAlpha):
                    m.alpha = alpha

    return ResNet18Alpha()


def build_resnet18_preact(num_classes=2):
    """
    Constructs a ResNet-18 v2 (pre-activation) architecture using PreActBasicBlock.
    """
    layers = [2, 2, 2, 2]  # Same block configuration as ResNet-18

    class ResNet18Preact(models.resnet.ResNet):
        def __init__(self):
            super().__init__(block=PreActBasicBlock, layers=layers, num_classes=num_classes)
            # Replace the final fully connected layer
            in_feat = self.fc.in_features
            self.fc = nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    return ResNet18Preact()


def build_resnet18_gated(num_classes=2):
    """
    Constructs a ResNet-18 architecture using BasicBlockGated (Gated Residual Blocks).
    """
    layers = [2, 2, 2, 2]  # Number of blocks in each of the four layers

    class ResNet18Gated(models.resnet.ResNet):
        def __init__(self):
            super().__init__(block=BasicBlockGated, layers=layers, num_classes=num_classes)
            # Replace the final fully connected layer
            in_feat = self.fc.in_features
            self.fc = nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    return ResNet18Gated()


# ==============================
# 4. Model Builder Function
# ==============================
def build_model(model_name='resnet18', num_classes=2, alpha=1.0):
    """
    Returns a model instance given the variant name:
      - 'resnet18'        : Standard ResNet-18 (post-activation, pretrained)
      - 'resnet18_alpha'  : ResNet-18 with BasicBlockAlpha (y = F(x) + alpha*x)
      - 'resnet18_preact' : ResNet-18 v2 (pre-activation BasicBlock)
      - 'resnet18_gated'  : ResNet-18 with Gated Residual Blocks
      - 'resnet34'        : Standard ResNet-34 (pretrained)
      - 'resnet50'        : Standard ResNet-50 (pretrained)
    """
    if model_name == 'resnet18':
        # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return build_resnet18_alpha(alpha=alpha, num_classes=num_classes)
    elif model_name == 'resnet18_alpha':
        return build_resnet18_alpha(alpha=alpha, num_classes=num_classes)
    elif model_name == 'resnet18_preact':
        return build_resnet18_preact(num_classes=num_classes)
    elif model_name == 'resnet18_gated':
        return build_resnet18_gated(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # For pretrained variants (resnet18, resnet34, resnet50):
    # Freeze conv1 parameters
    for param in model.conv1.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


# ==============================
# 5. Training & Evaluation Loop
# ==============================
def train_one(model, model_name, train_ld, test_ld, device, epochs=100, lr=1e-5, patience=20):
    """
    Trains the model for a specified number of epochs using Adam optimizer and
    ReduceLROnPlateau scheduler (monitoring test accuracy). Implements early stopping.
    Returns a dictionary containing training/validation history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=7, verbose=False
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


def plot_comparison(histories, metric, out_path):
    """
    Plots the given metric over epochs for all model variants stored in histories.
    Each entry in 'histories' is a dict mapping model_name -> list of metric values.
    """
    plt.figure()
    for name, h in histories.items():
        plt.plot(range(1, len(h[metric]) + 1), h[metric], label=name)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved comparison for {metric} to {out_path}")


# ==============================
# 6. Main Script to Compare Variants
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

    train_ld, test_ld = get_data_loaders(img_dir, csv_path, num_workers=0)

    # Compare four ResNet-18 variants:
    #  1) 'resnet18'        : original ResNet-18 (post-activation, pretrained)
    #  2) 'resnet18_alpha'  : ResNet-18 with BasicBlockAlpha (y = F(x) + alpha*x)
    #  3) 'resnet18_preact' : ResNet-18 v2 (pre-activation BasicBlock)
    #  4) 'resnet18_gated'  : ResNet-18 with Gated Residual Blocks
    variants = [
        'resnet18',
        'resnet18_alpha',
        'resnet18_preact',
        'resnet18_gated'
    ]
    histories = {}

    for v in variants:
        print(f"\n==> Training {v}...")

        if v == 'resnet18':
            model = build_model(model_name=v, num_classes=2, alpha=1)
        elif v == 'resnet18_alpha':
            # Use alpha=0.5 for the experiment; adjust as needed
            model = build_model(model_name=v, num_classes=2, alpha=0.7)
        elif v == 'resnet18_preact':
            model = build_model(model_name=v, num_classes=2)
        elif v == 'resnet18_gated':
            model = build_model(model_name=v, num_classes=2)
        else:
            raise ValueError(f"Unknown variant: {v}")

        histories[v] = train_one(model, v, train_ld, test_ld, device)

    # Plot all metrics for each variant
    for metric in histories[variants[0]].keys():
        plot_comparison(histories, metric, f'compare_{metric}.png')