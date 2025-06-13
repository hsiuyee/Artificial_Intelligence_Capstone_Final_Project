import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from vision_transformer_pytorch import VisionTransformer
import argparse

# ----------------------
# Argparser
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help='Path to GlaS@MICCAI dataset root')
parser.add_argument('--arch', default='ViT-B_16')
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--image-size', default=224, type=int)
parser.add_argument('--num-classes', default=2, type=int)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--logdir', default='runs/vit_glas')
parser.add_argument('--save-path', default='checkpoint_vit.pth', help='Path to save the best model')


# ----------------------
# Validation Function
# ----------------------
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total


# ----------------------
# Training Function
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    acc = 100. * correct / total
    return total_loss, acc


# ----------------------
# Main Function
# ----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    if args.pretrained:
        model = VisionTransformer.from_pretrained(args.arch, num_classes=args.num_classes)
    else:
        model = VisionTransformer.from_name(args.arch, num_classes=args.num_classes)
    model.to(device)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)

    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Dataset
    base    = os.path.dirname(__file__)
    img_dir = os.path.join(base, '../../Preprocess', 'Warwick_QU_Dataset')
    
    train_loader, val_loader = get_data_loaders(img_dir, csv_path= os.path.join(img_dir, 'Grade.csv'))

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    # train
    for epoch in range(args.epochs):
        total_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)

        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, args.save_path)
            print(f"✅ Saved new best model to {args.save_path} (Val Acc: {val_acc:.2f}%)")

    writer.close()


# ----------------------
# Entry Point
# ----------------------
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
