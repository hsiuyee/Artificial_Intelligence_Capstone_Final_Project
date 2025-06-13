import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from utils import FewShotSegContrastiveDataset  
from model import get_model
from loss import DiceLoss, pixel_contrastive_loss, BCEWithLogitsLoss
from utils import get_transform, dice_score


# --- Evaluation for segmentation ---
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)
            count += 1

    return val_loss / count, val_dice / count

# --- Train one epoch ---
def train_one_epoch(model, optimizer, train_loader, seg_loss_func, device, lambda_contrastive=0.1):
    model.train()
    running_loss = 0.0

    for img1, mask1, img2, mask2, label in train_loader:
        img1, img2 = img1.to(device), img2.to(device)
        mask1, mask2 = mask1.to(device), mask2.to(device)
        label = label.to(device)

        outputs1 = model(img1)
        seg_loss = seg_loss_func(outputs1, mask1)

        feat1 = model.encoder(img1)[-1]
        feat2 = model.encoder(img2)[-1]
        contrastive = pixel_contrastive_loss(feat1, feat2, label)

        total_loss = seg_loss + lambda_contrastive * contrastive

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    return running_loss

# --- Full training ---
def train(model, writer, device, train_loader, val_loader, args):
    bce = BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    num_epochs = 50
    best_dice = 0.0
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):
        model.train()

        running_loss = train_one_epoch(
            model, optimizer, train_loader,
            seg_loss_func=lambda p,t: bce(p,t)+dice(p,t),
            device=device,
            lambda_contrastive=0.1
        )

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = evaluate(model, val_loader, lambda p,t: bce(p,t)+dice(p,t), device)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = f"info/models/{args.model_name}_{args.shot}_shots.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated and saved at {save_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    writer.close()

# --- Main ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FPN", help="Model name: Unet, FPN, DeepLabV3+")
    parser.add_argument("--shot", type=int, default=10, help="Few-shot total number")
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Dataset root folder")
    args = parser.parse_args()

    os.makedirs("info/models", exist_ok=True)
    os.makedirs(f"info/logs/shot{args.shot}", exist_ok=True)

    writer = SummaryWriter(
        log_dir=f"info/logs/{args.model_name}_shot{args.shot}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        model_name=args.model_name,
        encoder_name="resnet50"
    ).to(device)

    train_dataset = FewShotSegContrastiveDataset(
        dataset_root=args.dataset_root,
        transform=get_transform(train=True),
        mode="train",
        k=args.shot
    )

    val_dataset = FewShotSegContrastiveDataset(
        dataset_root=args.dataset_root,
        transform=get_transform(train=False),
        mode="val"
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    train(model, writer, device, train_loader, val_loader, args)
