# project/test.py

import os
import torch
import argparse
import matplotlib.pyplot as plt
from model import get_model
from utils import get_transform
from PIL import Image
import numpy as np

def load_image(image_path, transform):
    image = np.array(Image.open(image_path).convert("RGB"))
    augmented = transform(image=image)
    image = augmented["image"].unsqueeze(0)  # (1, 3, H, W)
    return image

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)
        pred_mask = (output > 0.5).float()
    return pred_mask.squeeze().cpu().numpy()

def visualize(image_path, pred_mask, model_name, save_root="info/predictions"):
    # Make dir: info/predictions/Unet/
    save_dir = os.path.join(save_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    image = Image.open(image_path).convert("RGB")

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(pred_mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    # Save as info/predictions/Unet/testA_1_pred.png
    image_name = os.path.basename(image_path).replace(".bmp", "")
    save_path = os.path.join(save_dir, f"{image_name}_pred.png")
    plt.savefig(save_path)
    plt.close()


def test(model_name):
    encoder_name = "resnet50"
    model_path = f"info/models/{model_name}.pth"
    test_image_dir = "dataset/images/val"
    ground_truth_dir = "dataset/masks/val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name=model_name, encoder_name=encoder_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    transform = get_transform(train=False)
    test_images = os.listdir(test_image_dir)
    test_images = [os.path.join(test_image_dir, fname) for fname in test_images]
    count = 0
    max_outputs = 5

    for image_path in test_images:
        image_name = os.path.basename(image_path)
        mask_path = os.path.join(ground_truth_dir, image_name)

        if not os.path.exists(mask_path):
            continue

        # Load ground truth mask
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = (gt_mask > 0).astype(np.float32)

        # Skip if ground truth mask is empty
        if gt_mask.sum() == 0:
            continue

        # Predict and visualize
        image_tensor = load_image(image_path, transform)
        pred_mask = predict(model, image_tensor, device)
        visualize(image_path, pred_mask, model_name)
        count += 1

        if count >= max_outputs:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Unet", help="Choose model name (e.g., Unet, FPN, etc)")
    args = parser.parse_args()

    test(args.model_name)
