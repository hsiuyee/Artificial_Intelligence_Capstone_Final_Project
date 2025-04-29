import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# 1. Parameters
DATA_DIR = './Warwick_QU_Dataset'
SAVE_DIR = './Warwick_QU_Dataset_denoised'
os.makedirs(SAVE_DIR, exist_ok=True)
IMG_SIZE = 512  # Adjust as needed

# 2. Dataset definition
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.files = sorted(glob(os.path.join(root_dir, '*.bmp')))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        # Add noise
        noise = torch.randn_like(img) * 0.2   # Change 0.2 for desired noise strength
        noisy_img = (img + noise).clamp(0, 1)
        return noisy_img, img

# 3. Autoencoder definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 4. Prepare data and model
dataset = NoisyImageDataset(DATA_DIR, IMG_SIZE)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---- Device selection: prefer MPS (Apple Silicon GPU), then CUDA, then CPU ----
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Silicon GPU (MPS).")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU.")
else:
    device = torch.device('cpu')
    print("Using CPU.")

model = DenoisingAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 5. Train the model
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for noisy, clean in tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = loss_fn(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Average loss: {epoch_loss / len(dataloader):.4f}')

# 6. Denoise and save images
model.eval()
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
with torch.no_grad():
    files = sorted(glob(os.path.join(DATA_DIR, '*.bmp')))
    for f in tqdm(files, desc='Denoising and saving'):
        img = Image.open(f).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        output = model(img_tensor)[0].cpu().clamp(0,1)
        out_img = transforms.ToPILImage()(output)
        out_img.save(os.path.join(SAVE_DIR, os.path.basename(f)))

print('All done! Denoised images are saved in:', SAVE_DIR)
