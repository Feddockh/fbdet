import os
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from typing import Dict, List, Optional


class MultiCamModel(nn.Module):
    def __new__(cls, cameras: List[str], num_classes: int, model_type: str = "FasterRCNN",
                pretrained: bool = True):
        """
        Factory for multi-camera detection models. Currently supports:
        - FasterRCNN: early-fusion by stacking camera channels.

        Args:
            cameras: list of camera names
            num_classes: number of object classes (including background)
            model_type: type of model to create
            pretrained: whether to load pretrained weights
        """
        if model_type == "FasterRCNN":
            return MultiCamFasterRCNN(cameras, num_classes, pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class MultiCamFasterRCNN(nn.Module):
    def __init__(self, cameras: List[str], num_classes: int,
                 pretrained: bool = True):
        """
        A multi-camera Faster R-CNN model that uses early fusion by stacking
        image channels for N cameras such that input is [B, N*3, H, W].

        Args:
            cameras: list of camera names
            num_classes: number of object classes (including background)
            pretrained: whether to load pretrained weights
        """
        super().__init__()
        self.cameras = cameras

        # Load base model
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # Modify first conv to accept N*3 channels
        num_cam = len(cameras)
        backbone = model.backbone.body
        orig_conv = backbone.conv1
        in_channels = num_cam * 3
        new_conv = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=(orig_conv.bias is not None)
        )

        # Copy weights from original conv, repeat for N cameras
        with torch.no_grad():
            w = orig_conv.weight  # [out,3,k,k]
            new_w = w.repeat(1, num_cam, 1, 1) / float(num_cam)
            new_conv.weight.copy_(new_w)
            if orig_conv.bias is not None:
                new_conv.bias.copy_(orig_conv.bias)
        backbone.conv1 = new_conv

        # Replace predictor head to match num_classes
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = type(model.roi_heads.box_predictor)(in_feat, num_classes)

        self.model = model

    def forward(self, batch: Dict[str, List[torch.Tensor]],
                targets: Optional[Dict[str, List[Dict]]] = None):
        """
        Pass a batch of images through the model.

        Args:
            batch: dict mapping camera name to list of image tensors [C,H,W]
            targets: optional dict mapping camera name to list of target dicts for training

        Returns:
            If targets is provided (training): a dict of losses summed across cameras
            Else: a dict mapping camera name to list of predictions
        """
        # Fuse per-camera images
        cam0 = self.cameras[0]
        bs = len(batch[cam0])
        fused_imgs = []
        fused_tgts = []

        # Concatenate images along channel dim
        for i in range(bs):
            imgs = [batch[cam][i] for cam in self.cameras]
            fused = torch.cat(imgs, dim=0)
            fused_imgs.append(fused)

            # Concatenate targets if provided
            if targets is not None:
                all_boxes = []
                all_labels = []
                for cam in self.cameras:
                    t = targets[cam][i]

                    # Extract raw box tensor from BoundingBoxes if needed
                    boxes = t['boxes'].tensor if hasattr(t['boxes'], 'tensor') else t['boxes']
                    labels = t['labels']
                    all_boxes.append(boxes)
                    all_labels.append(labels)
                fused_boxes = torch.cat(all_boxes, dim=0)
                fused_labels = torch.cat(all_labels, dim=0)
                fused_tgts.append({'boxes': fused_boxes, 'labels': fused_labels})

        if self.training:
            return self.model(fused_imgs, fused_tgts)
        else:
            return self.model(fused_imgs)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_type': 'FasterRCNN',
            'cameras': self.cameras,
            'state_dict': self.state_dict()
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, num_classes: int, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        cameras = ckpt['cameras']
        model = cls(cameras, num_classes, pretrained=False)
        model.load_state_dict(ckpt['state_dict'])
        return model



