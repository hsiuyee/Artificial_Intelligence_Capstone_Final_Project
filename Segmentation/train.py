import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from model import get_model, get_loss
from utils import get_transform, dice_score, get_dataloader
from segmentation_models_pytorch.metrics import iou_score, f1_score


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    # val_iou = 0.0
    # val_f1 = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)
            # val_iou += iou_score(outputs, masks)
            # val_f1  += f1_score(outputs, masks)
            count += 1

    return (
        val_loss / count,
        val_dice / count,
        # val_iou  / count,
        # val_f1   / count,
    )


def train_one_epoch(model, optimizer, train_loader, criterion, device, lambda_align=1.0):
    model.train()
    running_loss = 0.0

    # Standard full-supervision training
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss


def train(model, writer, device, train_loader, val_loader, args):

    criterion = get_loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    num_epochs = 50
    best_dice = 0.0
    early_stop_counter = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):
        model.train()

        running_loss = train_one_epoch(model, optimizer, train_loader, criterion, device)
        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        # writer.add_scalar("IOU/val", val_iou, epoch)
        # writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = f"info/models/{args.model_name}.pth"
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
    

if __name__ == "__main__":
    os.makedirs("info/models", exist_ok=True)
    os.makedirs("info/logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FPN", help="Model name: Unet, FPN, DeepLabV3+")
    args = parser.parse_args()

    writer = SummaryWriter(
        log_dir=f"info/logs/{args.model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        model_name=args.model_name,
        encoder_name="resnet50"
    ).to(device)

    train_loader, val_loader = get_dataloader()

    train(model, writer, device, train_loader, val_loader, args)


