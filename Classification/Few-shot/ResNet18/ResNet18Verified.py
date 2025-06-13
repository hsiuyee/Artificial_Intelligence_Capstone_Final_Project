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
from sklearn.metrics import f1_score, roc_curve, auc

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
        tc = 0
        name = 0
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
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        else:
            idx1 = idx // 10
            idx2 = idx % 10
            tc = idx2 % 2
            base1 = self.img_names[idx1]
            name = base1
            path = os.path.join(self.img_dir, base1 + '.bmp')
            img1 = Image.open(path).convert('RGB')
            base2 = self.positive[idx2 // 2] if idx2 % 2 else self.negative[idx2 // 2]
            path = os.path.join(self.img_dir, base2 + '.bmp')
            img2 = Image.open(path).convert('RGB')
            lab = 0 if self.labels[base1] == idx % 2 else 1
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        return img1, img2, lab, tc, name

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
    return train_ld, test_ld, test_ds

# --- Load a pretrained AlexNet and replace its classifier ---
def build_model():

    model = models.resnet18()
    model.load_state_dict(torch.load("./Classification/Few-shot/ResNet18/Res18_fewshot.pth", weights_only=True))
    return model

def valid(model, train_ld, test_ld, device, test_ds):
    best_acc = 0
    best_thr = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for x1, x2, y, _, _ in tqdm(train_ld):
            x1, x2 = x1.to(device), x2.to(device)
            dist = torch.nn.functional.pairwise_distance(model(x1), model(x2), keepdim=False)
            thr = dist.cpu().numpy()
            for i in range(44):
                margin = (thr[i] + thr[i+1]) / 2
                preds = (dist >= margin).cpu().numpy()
                tr = (preds == y).cpu().numpy().copy()
                tr.dtype = np.int8
                test_acc = tr.mean()
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_thr = margin
                if test_acc == best_acc:
                    best_thr = best_thr if abs(best_thr - 2) < abs(margin - 2) else margin
                print(test_acc, end=' ')
    all_tc = []
    all_name = []
    all_dist = []
    with torch.no_grad():
        for x1, x2, y, tc, name in tqdm(test_ld):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            dist = torch.nn.functional.pairwise_distance(model(x1), model(x2), keepdim=False)
            preds = (dist >= best_thr)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_dist.append(dist.cpu())
            all_tc.extend(tc)
            all_name.extend(name)
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_dist = torch.cat(all_dist).numpy()

    pred_cnt = dict()
    for i in range(len(all_tc)):
        name = all_name[i]
        if name not in pred_cnt.keys():
            pred_cnt[name] = [0, 0]
        if all_preds[i]:
            pred_cnt[name][1 - all_tc[i]] += 1
        else:
            pred_cnt[name][all_tc[i]] += 1

    confusion = [[0, 0], [0, 0]]
    summ = 0
    conf = 0
    for k, pl in pred_cnt.items():
        summ += 1
        tc = test_ds.labels[k]
        if pl[1] == pl[0]:
            pc = -1
            conf += 1
        else:
            pc = 1 if pl[1] > pl[0] else 0
        if pc >= 0:
            confusion[tc][pc] += 1
    real_acc = (confusion[0][0] + confusion[1][1]) / summ

    test_acc = (all_preds == all_labels).mean()
    test_f1  = f1_score(all_labels, all_preds, average='binary')

    fpr, tpr, _ = roc_curve(all_labels, all_dist)
    with open("./Classification/Few-shot/ResNet18/roc_record.txt", "w") as file:
        for n in fpr:
            file.write(f"{n}, ")
        file.write("\n")
        for n in tpr:
            file.write(f"{n}, ")
        file.write("\n")
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./Classification/Few-shot/ResNet18/ROC.png")
    print(f"ACC: {test_acc*100}%, F1: {test_f1}, AUC: {roc_auc}, Real ACC: {real_acc*100}%")
    print(f"Confusion:\n \
          {confusion[0][0]*100/summ}%\t{confusion[0][1]*100/summ}%\n \
          {confusion[1][0]*100/summ}%\t{confusion[1][1]*100/summ}%")
    print(f"unknown: {conf}, {conf*100/summ}%")

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

    train_ld, test_ld, test_ds = get_data_loaders(img_dir, csv_path)
    model = build_model().to(device)

    try:
        valid(model, train_ld, test_ld, device, test_ds)
    except KeyboardInterrupt:
        print("KeyboardInturrupt")
        if device == 'cuda':
            torch.cuda.empty_cache()
        os._exit(1)
