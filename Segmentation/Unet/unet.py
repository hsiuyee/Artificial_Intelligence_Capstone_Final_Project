import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Custom Segmentation Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # data_dir contains .bmp images and corresponding *_anno.bmp masks
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bmp') and '_anno' not in f])
        self.img_paths = [os.path.join(data_dir, f) for f in files]
        self.mask_paths = [os.path.join(data_dir, f.replace('.bmp','_anno.bmp')) for f in files]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            # apply same resize to mask using PIL
            image = self.transform(image)
            # resize mask to match image size
            _, h, w = image.shape
            mask = mask.resize((w, h), resample=Image.NEAREST)
            mask = transforms.ToTensor()(mask)
        return image, mask

# --- Data loaders splitting by filename containing 'train' or 'test' ---
def get_dataloaders(data_dir, batch_size=4, input_size=512):
    # gather train/test based on filename
    all_images = sorted([f for f in os.listdir(data_dir) if f.endswith('.bmp') and '_anno' not in f])
    train_files = [f for f in all_images if 'train' in f.lower()]
    test_files  = [f for f in all_images if 'test' in f.lower()]

    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ])

    # create datasets
    train_ds = SegmentationDataset(data_dir, transform)
    test_ds  = SegmentationDataset(data_dir, transform)
    train_ds.img_paths = [os.path.join(data_dir,f) for f in train_files]
    train_ds.mask_paths = [os.path.join(data_dir,f.replace('.bmp','_anno.bmp')) for f in train_files]
    test_ds.img_paths  = [os.path.join(data_dir,f) for f in test_files]
    test_ds.mask_paths = [os.path.join(data_dir,f.replace('.bmp','_anno.bmp')) for f in test_files]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,num_workers=4)
    return train_loader, test_loader

# --- AtResUNet Model ---
class AtResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        def atrous_block(ch):
            return nn.Sequential(
                *[nn.Conv2d(ch, ch, 3, padding=r, dilation=r) for r in [1,2,4,8,16,32]]
            )

        # Encoder
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Center
        self.center = conv_block(512, 1024)
        # Decoder
        self.dec4 = conv_block(1024 + 512, 512)
        self.dec3 = conv_block(512 + 256, 256)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)
        # Atrous residual modules
        self.atrous4 = atrous_block(512)
        self.atrous3 = atrous_block(256)
        self.atrous2 = atrous_block(128)
        self.atrous1 = atrous_block(64)
        # Final conv
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c  = self.center(self.pool(e4))

        # Decoder with Atrous residual
        d4_in = torch.cat([nn.functional.interpolate(c, scale_factor=2, mode='bilinear', align_corners=True), e4], dim=1)
        d4_out = self.dec4(d4_in)
        d4 = d4_out + self.atrous4(d4_out)

        d3_in = torch.cat([nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1)
        d3_out = self.dec3(d3_in)
        d3 = d3_out + self.atrous3(d3_out)

        d2_in = torch.cat([nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1)
        d2_out = self.dec2(d2_in)
        d2 = d2_out + self.atrous2(d2_out)

        d1_in = torch.cat([nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1)
        d1_out = self.dec1(d1_in)
        d1 = d1_out + self.atrous1(d1_out)

        return torch.sigmoid(self.final(d1))

# --- Training ---
def train_model(model, train_loader, test_loader, device, epochs=50, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    def dice_loss(pred,gt,smooth=1):
        pred,gt = pred.view(-1), gt.view(-1)
        inter = (pred*gt).sum()
        return 1 - (2*inter+smooth)/(pred.sum()+gt.sum()+smooth)
    for ep in range(1,epochs+1):
        model.train()
        train_loss=0
        for img,mask in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} [Train]"):
            img,mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss= dice_loss(out,mask)
            loss.backward(); optimizer.step()
            train_loss+= loss.item()
        print(f"Epoch {ep} Train Loss: {train_loss/len(train_loader):.4f}")
        # validation
        model.eval()
        val_loss=0
        with torch.no_grad():
            for img,mask in tqdm(test_loader, desc=f"Epoch {ep}/{epochs} [Val  ]"):
                img,mask = img.to(device), mask.to(device)
                out = model(img)
                val_loss+= dice_loss(out,mask).item()
        print(f"Epoch {ep} Val Loss:   {val_loss/len(test_loader):.4f}")

# --- Main ---
if __name__=='__main__':
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir,'..','..','Preprocess','Warwick_QU_Dataset'))
    train_loader, test_loader = get_dataloaders(data_dir)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using {device}")
    model = AtResUNet()
    train_model(model, train_loader, test_loader, device)
