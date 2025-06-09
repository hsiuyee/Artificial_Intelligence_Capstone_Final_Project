import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_curve, auc  # add this import

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
        self.negative = ('train_2', 'train_15', 'train_12', 'train_20', 'train_21')
        self.positive = ('train_1', 'train_10', 'train_11', 'train_13', 'train_14')
        self.transform = transform
        self.randomAugment = transforms.RandAugment(1, 19)
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
            if 'anno' in base or 'train' in base:
                continue
            path = os.path.join(img_dir, base + '.bmp')
            if os.path.exists(path):
                self.img_names.append(base)

        if not self.img_names:
            raise FileNotFoundError(f"No .bmp files for split='{split}' in {img_dir}")

    def __len__(self):
        if self.type == 'train':
            return 45
        else:
            return len(self.img_names) * 10

    def __getitem__(self, idx):
        lab = 0 if idx < 20 else 1
        tmp = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        if self.type == 'train':
            if idx < 20:
                idx1, idx2 = tmp[idx % 2]
                base1 = self.positive[idx1] if idx <10 else self.negative[idx1]
                path = os.path.join(self.img_dir, base1 + '.bmp')
                img1 = Image.open(path).convert('RGB')
                base2 = self.positive[idx2] if idx <10 else self.negative[idx2]
                path = os.path.join(self.img_dir, base2 + '.bmp')
                img2 = Image.open(path).convert('RGB')
            else:
                idx -= 20
                idx1 = idx // 5
                idx2 = idx % 5
                base1 = self.positive[idx1]
                path = os.path.join(self.img_dir, base1 + '.bmp')
                img1 = Image.open(path).convert('RGB')
                base2 = self.negative[idx2]
                path = os.path.join(self.img_dir, base2 + '.bmp')
                img2 = Image.open(path).convert('RGB')
            if self.transform:
                img1 = self.randomAugment(img1)
                img1 = self.transform(img1)
                img2 = self.randomAugment(img2)
                img2 = self.transform(img2)
        else:
            idx1 = idx // 10
            idx2 = idx % 10
            base1 = self.img_names[idx1]
            path = os.path.join(self.img_dir, base1 + '.bmp')
            img1 = Image.open(path).convert('RGB')
            base2 = self.positive[idx2 // 2] if idx2 % 2 else self.negative[idx2 // 2]
            path = os.path.join(self.img_dir, base2 + '.bmp')
            img2 = Image.open(path).convert('RGB')
            lab = 0 if self.labels[base1] == idx % 2 else 1
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        return img1, img2, lab

# --- Build DataLoaders for train & test sets ---
def get_data_loaders(img_dir, csv_path,
                     input_size=224, batch_size=64,
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
def criterion(x1, x2, label, margin: float = 0.5):
    """
    Computes Contrastive Loss
    """
    dist = torch.nn.functional.pairwise_distance(x1, x2)
    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss

# --- Load a pretrained AlexNet and replace its classifier ---
def build_model():

    model = models.alexnet()
    model.load_state_dict(torch.load("./Classification/Few-shot/AlexNet/Alex_fewshot_aug.pth", weights_only=True))
    return model

def valid(model, train_ld, test_ld, device):
    best_acc = 0
    best_thr = 0
    all_preds = []
    all_labels = []


    model.eval()
    with torch.no_grad():
        for x1, x2, y in tqdm(train_ld):
            x1, x2 = x1.to(device), x2.to(device)
            dist = torch.nn.functional.pairwise_distance(model(x1), model(x2), keepdim=False)
            thr = dist.cpu().numpy()
            for i in range(19):
                margin = (thr[i] + thr[i+1]) / 2
                preds = (dist >= margin).cpu().numpy()
                tr = (preds == y).cpu().numpy().copy()
                tr.dtype = np.int8
                test_acc = tr.mean()
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_thr = margin
                if test_acc == best_acc:
                    best_thr = best_thr if abs(best_thr - 0.25) < abs(margin - 0.25) else margin
                print(test_acc)

    with torch.no_grad():
        for x1, x2, y in tqdm(test_ld):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            dist = torch.nn.functional.pairwise_distance(model(x1), model(x2), keepdim=False)
            preds = (dist >= best_thr)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    test_acc = (all_preds == all_labels).mean()
    test_f1  = f1_score(all_labels, all_preds, average='binary')

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./Classification/Few-shot/AlexNet/ROC_aug.png")
    print(f"ACC: {test_acc}, F1: {test_f1}, AUC: {roc_auc}")

    return

# --- Main entrypoint ---
if __name__ == '__main__':
    base    = os.path.dirname(__file__)
    img_dir = os.path.join(base, '../../../Preprocess', 'Warwick_QU_Dataset')
    csv_path= os.path.join(img_dir, 'Grade.csv')

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("Using device:", device)

    train_ld, test_ld = get_data_loaders(img_dir, csv_path)
    model = build_model().to(device)

    try:
        valid(model, train_ld, test_ld, device)
    except KeyboardInterrupt:
        print("KeyboardInturrupt")
        if device == 'cuda':
            torch.cuda.empty_cache()
        os._exit(1)
