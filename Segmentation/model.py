# project/model.py

import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(model_name="Unet", encoder_name="resnet34", in_channels=3, classes=1):
    """Create segmentation model dynamically."""
    
    # Choose model dynamically
    if model_name == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    elif model_name == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    elif model_name == "DeepLabV3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

def get_loss():
    return nn.BCEWithLogitsLoss()
