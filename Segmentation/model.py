import segmentation_models_pytorch as smp
import torch.nn as nn

def get_model(model_name="Unet", encoder_name="resnet34", in_channels=3, classes=1):
    model_dict = {
        "Unet": smp.Unet,
        "Unet++": smp.UnetPlusPlus,
        "MAnet": smp.MAnet,
        "Linknet": smp.Linknet,
        "FPN": smp.FPN,
        "PSPNet": smp.PSPNet,
        "PAN": smp.PAN,
        "DeepLabV3": smp.DeepLabV3,
        "DeepLabV3+": smp.DeepLabV3Plus,
    }

    if model_name not in model_dict:
        raise ValueError(f" Model '{model_name}' not supported.\nChoose from: {list(model_dict.keys())}")

    model_class = model_dict[model_name]
    return model_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
        activation=None
    )

def get_loss():
    return nn.BCEWithLogitsLoss()
