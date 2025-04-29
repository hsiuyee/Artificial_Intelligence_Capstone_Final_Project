import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from model import get_model, get_loss
from utils import get_transform, dice_score, GlaSDataset

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)

    avg_loss = val_loss / len(val_loader)
    avg_dice = val_dice / len(val_loader)

    return avg_loss, avg_dice

def train(model_name):
    # 1. Folders
    os.makedirs("info/models", exist_ok=True)
    os.makedirs("info/logs", exist_ok=True)

    # 2. Tensorboard
    writer = SummaryWriter(log_dir="info/logs")

    # 3. Data preparation
    train_dataset = GlaSDataset("dataset/images/train", "dataset/masks/train", transform=get_transform(train=True))
    val_dataset = GlaSDataset("dataset/images/val", "dataset/masks/val", transform=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    # 4. Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        model_name=model_name,
        encoder_name="resnet50",
        in_channels=3,
        classes=1
    ).to(device)

    criterion = get_loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # 5. Training loop
    num_epochs = 100
    best_dice = 0.0
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # 🔥 Evaluate after each epoch
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # 🔥 Tensorboard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # 🔥 Learning rate scheduler
        scheduler.step(val_dice)

        # 🔥 Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = f"info/models/best_{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated and saved at {save_path}")
            early_stop_counter = 0  # Reset counter if improved
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{early_stop_patience}")

        # 🔥 Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FPN", help="Model name: Unet, FPN, DeepLabV3+")
    args = parser.parse_args()

    train(args.model_name)
