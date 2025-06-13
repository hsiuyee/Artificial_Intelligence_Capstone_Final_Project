import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def get_model(model_name="Unet", encoder_name="resnet34", in_channels=3, classes=1, have_shot=False):
    if not have_shot:
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
            raise ValueError(f"Model '{model_name}' not supported.\nChoose from: {list(model_dict.keys())}")

        model_class = model_dict[model_name]
        return model_class(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            activation=None
        )
    else:
        return KShotSiameseSegNet(
            model_arch=model_name,  # 修正這邊使用正確的參數名稱
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            out_channels=classes,
        )

def get_loss():
    return nn.BCEWithLogitsLoss()


class KShotSiameseSegNet(nn.Module):
    def __init__(self, model_arch='Unet', encoder_name='resnet34', encoder_weights='imagenet', out_channels=1):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, weights=encoder_weights)
        encoder_channels = self.encoder.out_channels[::-1]  # reverse for decoder

        if model_arch.lower() == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=[encoder_channels[0]*2] + encoder_channels[1:],
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_norm="batchnorm",
            )
            seg_out_channels = 16

        elif model_arch.lower() == 'fpn':
            self.decoder = smp.decoders.fpn.decoder.FPNDecoder(
                encoder_channels=[encoder_channels[0]*2] + encoder_channels[1:],
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.1,
                merge_policy="add",
            )
            seg_out_channels = 128

        elif model_arch.lower() == 'manet':
            self.decoder = smp.decoders.manet.decoder.MANetDecoder(
                encoder_channels=[encoder_channels[0]*2] + encoder_channels[1:],
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_norm="batchnorm",
                center=False
            )
            seg_out_channels = 16

        elif model_arch.lower() == 'linknet':
            self.decoder = smp.decoders.linknet.decoder.LinknetDecoder(
                encoder_channels=[encoder_channels[0]*2] + encoder_channels[1:],
                decoder_channels=(256, 128, 64, 32),
                n_blocks=4,
                use_norm="batchnorm",
            )
            seg_out_channels = 32

        elif model_arch.lower() == 'pspnet':
            self.decoder = smp.decoders.pspnet.decoder.PSPDecoder(
                encoder_channels=encoder_channels,
                pyramid_channels=256,
                dropout=0.1,
            )
            seg_out_channels = 256

        elif model_arch.lower() == 'pan':
            self.decoder = smp.decoders.pan.decoder.PANDecoder(
                encoder_channels=encoder_channels,
                pyramid_channels=256,
                segmentation_channels=128,
            )
            seg_out_channels = 128

        elif model_arch.lower() == 'deeplabv3':
            self.decoder = smp.decoders.deeplabv3.decoder.DeepLabV3Decoder(
                encoder_channels=encoder_channels,
                atrous_rates=(12, 24, 36),
                output_stride=16,
            )
            seg_out_channels = 256

        elif model_arch.lower() == 'deeplabv3+':
            self.decoder = smp.decoders.deeplabv3plus.decoder.DeepLabV3PlusDecoder(
                encoder_channels=encoder_channels,
                atrous_rates=(12, 24, 36),
                output_stride=16,
            )
            seg_out_channels = 256

        else:
            raise ValueError(f"Unsupported model architecture: {model_arch}")

        self.seg_head = nn.Conv2d(seg_out_channels, out_channels, kernel_size=1)

    def masked_average(self, features, masks):
        masked = features * masks
        proto = masked.sum(dim=(2, 3)) / (masks.sum(dim=(2, 3)) + 1e-6)
        proto = proto.mean(dim=0)  # average over k shots
        return proto.view(1, -1, 1, 1)

    # def compute_alignment_loss(self, query_feat, pred_mask, support_feats, support_gt_masks):
    #     pred_mask = pred_mask.argmax(dim=1).unsqueeze(1).float()
    #     query_proto = self.masked_average(query_feat, pred_mask)
    #     loss = 0.0
    #     for sf, sm in zip(support_feats, support_gt_masks):
    #         sf = sf.unsqueeze(0)
    #         sm_resized = F.interpolate(sm.unsqueeze(0), size=sf.shape[-2:], mode='nearest')
    #         score = F.cosine_similarity(sf, query_proto.expand_as(sf), dim=1)
    #         loss += F.binary_cross_entropy_with_logits(score, sm_resized.squeeze(1))
    #     return loss / len(support_feats)

    def forward(self, support_imgs, support_masks, query_img, query_mask=None):
        k = support_imgs.size(0)
        support_feats = [self.encoder(support_imgs[i:i+1])[-1] for i in range(k)]
        support_feats = torch.cat(support_feats, dim=0)
        support_masks_resized = F.interpolate(support_masks, size=support_feats.shape[-2:], mode='nearest')

        prototype = self.masked_average(support_feats, support_masks_resized)

        query_feats = self.encoder(query_img)  # [stage1, stage2, ..., stage5]

        # 取最深的作為 top
        top_feat = query_feats[-1]

        # 與 prototype 拼接
        merged = torch.cat([top_feat, prototype.expand_as(top_feat)], dim=1)

        # 解碼器輸入：先拼接層，接 shallow features（反過來）
        decoder_input = [merged] + query_feats[:-1][::-1]  # e.g., [merged, stage4, stage3, stage2, stage1]

        # decode
        x = self.decoder(decoder_input)
        logits = self.seg_head(x)

        if self.training and query_mask is not None:
            query_top_feat = top_feat
            # align_loss = self.compute_alignment_loss(query_top_feat, logits, support_feats, support_masks)
            return logits
        else:
            return logits