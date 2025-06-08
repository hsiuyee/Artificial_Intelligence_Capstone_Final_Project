import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from model import get_model, get_loss
from utils import get_transform, dice_score, get_dataloader
from segmentation_models_pytorch.metrics import iou_score, f1_score


def evaluate(model, val_loader, criterion, device, shot):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    val_f1 = 0.0

    with torch.no_grad():
        if shot == -1:
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)
                val_iou += iou_score(outputs, masks)
                val_f1 += f1_score(outputs, masks)
        else:
            for support_img, support_mask, query_img, query_mask in val_loader:
                support_img = support_img.to(device)
                support_mask = support_mask.to(device)
                query_img = query_img.to(device)
                query_mask = query_mask.to(device)

                outputs = model(support_img, support_mask, query_img)
                val_loss += criterion(outputs, query_mask).item()
                val_dice += dice_score(outputs, query_mask)
                val_iou += iou_score(outputs, query_mask)
                val_f1  += f1_score(outputs, query_mask)

    # return avg
    return (
        val_loss / len(val_loader),
        val_dice / len(val_loader),
        val_iou / len(val_loader),
        val_f1 / len(val_loader),
    )


def train_one_epoch(shot, model, optimizer, train_loader):

    running_loss = 0.0
    if shot != -1:

        for support_imgs, support_masks, query_imgs, query_masks in train_loader:
            # Stack tensors (turn list of tensors into a batch tensor)
            support_imgs = torch.stack(support_imgs).to(device)       # [B, shot, 3, H, W]
            support_masks = torch.stack(support_masks).to(device)     # [B, shot, 1, H, W]
            query_imgs = torch.stack(query_imgs).to(device)           # [B, 3, H, W]
            query_masks = torch.stack(query_masks).to(device)         # [B, 1, H, W]

            outputs = model(support_imgs, support_masks, query_imgs)
            loss = criterion(outputs, query_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    else:
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

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

        running_loss = train_one_epoch(args.shot, model, optimizer, train_loader)
        train_loss = running_loss / len(train_loader)
        val_loss, val_dice, val_iou, val_f1 = evaluate(model, val_loader, criterion, device, args.shot)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IOU: {val_iou:.4f} | Val F1: {val_f1:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("IOU/val", val_iou, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
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
    

if __name__ == "__main__":
    os.makedirs("info/models", exist_ok=True)
    os.makedirs("info/logs", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FPN", help="Model name: Unet, FPN, DeepLabV3+")
    parser.add_argument("--shot", type=int, default=-1, help="Number of support shots (-1 means non-few-shot)")
    args = parser.parse_args()

    writer = SummaryWriter(
        log_dir=f"info/logs/{args.model_name}_shot{args.shot}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        model_name=args.model_name,
        encoder_name="resnet50",
        have_shot=args.shot != -1
    ).to(device)

    train_loader, val_loader = get_dataloader(args.shot)

    train(model, writer, device, train_loader, val_loader, args)


