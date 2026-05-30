from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


TIMM_FALLBACKS: dict[str, list[str]] = {
    "resnet50.a1_in1k": ["resnet50.a1_in1k", "resnet50.tv_in1k", "resnet50"],
    "tf_efficientnetv2_s.in21k_ft_in1k": ["tf_efficientnetv2_s.in21k_ft_in1k", "tf_efficientnetv2_s", "efficientnetv2_rw_s.ra2_in1k"],
    "convnextv2_tiny.fcmae_ft_in22k_in1k": ["convnextv2_tiny.fcmae_ft_in22k_in1k", "convnextv2_tiny", "convnext_tiny.fb_in22k_ft_in1k"],
    "mobilenetv3_large_100.ra_in1k": ["mobilenetv3_large_100.ra_in1k", "mobilenetv3_large_100"],
    "tf_efficientnet_b4.ns_jft_in1k": ["tf_efficientnet_b4.ns_jft_in1k", "efficientnet_b4.ra2_in1k", "efficientnet_b4"],
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": [
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "vit_base_patch16_224",
    ],
    "swin_tiny_patch4_window7_224.ms_in1k": ["swin_tiny_patch4_window7_224.ms_in1k", "swin_tiny_patch4_window7_224"],
    "deit3_small_patch16_224.fb_in22k_ft_in1k": ["deit3_small_patch16_224.fb_in22k_ft_in1k", "deit3_small_patch16_224"],
    "maxvit_tiny_tf_224.in1k": ["maxvit_tiny_tf_224.in1k", "maxvit_tiny_tf_224"],
    "vit_base_patch14_dinov2.lvd142m": ["vit_base_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2"],
}


@dataclass(frozen=True)
class CreatedModel:
    model: nn.Module
    resolved_name: str
    metadata: dict[str, Any]


def _candidate_names(model_name: str) -> list[str]:
    return TIMM_FALLBACKS.get(model_name, [model_name])


def _freeze_for_linear_probe(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    heads: list[nn.Module] = []
    for attr in ("head", "classifier", "fc"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            heads.append(module)
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            heads.append(classifier)
    trainable = 0
    for head in heads:
        for parameter in head.parameters():
            parameter.requires_grad = True
            trainable += parameter.numel()
    if trainable == 0:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                for parameter in module.parameters():
                    parameter.requires_grad = True
                return


def create_timm_head_model(
    method_cfg: dict[str, Any],
    num_outputs: int,
    pretrained: bool,
) -> CreatedModel:
    import timm

    requested = str(method_cfg["model_name"])
    errors: list[str] = []
    for name in _candidate_names(requested):
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=num_outputs)
            if bool(method_cfg.get("linear_probe", False)):
                _freeze_for_linear_probe(model)
            return CreatedModel(
                model=model,
                resolved_name=name,
                metadata={
                    "parameters": count_parameters(model)["parameters"],
                    "trainable_parameters": count_parameters(model)["trainable_parameters"],
                    "linear_probe": bool(method_cfg.get("linear_probe", False)),
                },
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"Could not create timm model {requested}. Errors:\n" + "\n".join(errors))


class TimmFPN(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool, decoder_channels: int = 128):
        super().__init__()
        import timm

        errors: list[str] = []
        for name in _candidate_names(model_name):
            try:
                self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
                self.resolved_name = name
                break
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        else:
            raise RuntimeError(f"Could not create timm FPN encoder {model_name}. Errors:\n" + "\n".join(errors))

        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(channel, decoder_channels, kernel_size=1) for channel in channels])
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels * len(channels), decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def _nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.ndim == 4 and feature.shape[1] != expected_channels and feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        target_size = features[0].shape[-2:]
        projected = []
        for feature, conv in zip(features, self.lateral):
            feature = self._nchw(feature, conv.in_channels)
            projected.append(F.interpolate(conv(feature), size=target_size, mode="bilinear", align_corners=False))
        logits = self.head(torch.cat(projected, dim=1))
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"out": logits}


def create_segmentation_model(method_cfg: dict[str, Any], num_classes: int, pretrained: bool) -> CreatedModel:
    framework = str(method_cfg.get("framework", "torchvision"))
    model_name = str(method_cfg["model_name"])
    if framework == "timm_fpn":
        model = TimmFPN(model_name, num_classes=num_classes, pretrained=pretrained)
        return CreatedModel(model=model, resolved_name=f"timm_fpn:{model.resolved_name}", metadata=count_parameters(model))

    import torchvision.models.segmentation as segmentation

    factory = getattr(segmentation, model_name)
    kwargs: dict[str, Any] = {"weights": None, "num_classes": num_classes}
    if not pretrained:
        kwargs["weights_backbone"] = None
    model = factory(**kwargs)
    return CreatedModel(model=model, resolved_name=f"torchvision:{model_name}", metadata=count_parameters(model))


def create_detection_model(method_cfg: dict[str, Any], num_classes_with_background: int, pretrained: bool) -> CreatedModel:
    import torchvision.models.detection as detection

    model_name = str(method_cfg["model_name"])
    factory = getattr(detection, model_name)
    kwargs: dict[str, Any] = {"weights": None, "num_classes": num_classes_with_background}
    if not pretrained:
        kwargs["weights_backbone"] = None
    model = factory(**kwargs)
    return CreatedModel(model=model, resolved_name=f"torchvision:{model_name}", metadata=count_parameters(model))


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}

