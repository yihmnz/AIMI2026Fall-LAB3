"""Shared dataset, preprocessing and model utilities for LAB3 (4-class CXR)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

CLASSES = ["normal", "bacteria", "virus", "COVID-19"]
MODEL_CHOICES = ["imagenet_densenet", "xrv_densenet", "rad_dino", "fusion"]
RAD_DINO_ID = "microsoft/rad-dino"


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def csv_path(data_root: Path, name: str) -> Path:
    for candidate in (data_root / name, data_root.parent / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {name}. Looked under {data_root} and {data_root.parent}.")


def read_labels(path: Path, require_labels: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"new_filename": "string"})
    required = ["new_filename"] + CLASSES
    if frame.columns.tolist() != required:
        raise ValueError(f"CSV columns must be exactly {required}; got {frame.columns.tolist()}")
    if frame["new_filename"].isna().any() or frame["new_filename"].duplicated().any():
        raise ValueError(f"Missing or duplicated new_filename in {path}")
    if require_labels:
        labels = frame[CLASSES].to_numpy(dtype=np.float64)
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Labels must be 0/1 with no missing values: {path}")
        if not np.all(labels.sum(axis=1) == 1):
            raise ValueError(f"Each row must have exactly one positive label: {path}")
    return frame


def rad_norm_stats() -> tuple[list[float], list[float]]:
    """Read normalization metadata rather than silently using ImageNet statistics."""
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(RAD_DINO_ID)
    return list(proc.image_mean), list(proc.image_std)


class CXRDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        frame: pd.DataFrame,
        model_type: str,
        image_size: int = 224,
        augment: bool = False,
        rad_stats: Optional[tuple[list[float], list[float]]] = None,
        labeled: bool = True,
        check_files: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.frame = frame.reset_index(drop=True)
        self.model_type = model_type
        self.labeled = labeled
        if check_files:
            missing = [n for n in self.frame["new_filename"] if not (self.image_dir / str(n)).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} images not found in {self.image_dir}; examples: {missing[:5]}"
                )
        self.targets = self.frame[CLASSES].to_numpy(dtype=np.int64).argmax(axis=1) if labeled else None
        self.augment = T.Compose([
            T.RandomRotation(5, fill=0),
            T.RandomApply([T.ColorJitter(brightness=0.08, contrast=0.08)], p=0.50),
        ]) if augment else None
        self.crop = T.Compose([
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
        ])
        self.to_tensor = T.ToTensor()
        if model_type in {"rad_dino", "fusion"}:
            if rad_stats is None:
                rad_stats = rad_norm_stats()
            self.rad_normalize = T.Normalize(*rad_stats)
        if model_type == "imagenet_densenet":
            self.imagenet_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        path = self.image_dir / str(self.frame.iloc[index]["new_filename"])
        with Image.open(path) as raw:
            image = raw.convert("L")  # Uniform grayscale input; ignore source file format.
            if self.augment is not None:
                image = self.augment(image)
            image = self.crop(image)
            pixels = self.to_tensor(image)  # (1, H, W), in [0, 1]
        if self.model_type == "xrv_densenet":
            # xrv.datasets.normalize(x, 255): convert [0,255] -> [-1024,1024]
            data = (pixels * 2.0 - 1.0) * 1024.0
        elif self.model_type == "imagenet_densenet":
            data = self.imagenet_normalize(pixels.repeat(3, 1, 1))
        elif self.model_type == "rad_dino":
            data = self.rad_normalize(pixels.repeat(3, 1, 1))
        elif self.model_type == "fusion":
            data = {
                "xrv": (pixels * 2.0 - 1.0) * 1024.0,
                "rad": self.rad_normalize(pixels.repeat(3, 1, 1)),
            }
        else:
            raise ValueError(self.model_type)
        if self.labeled:
            return data, int(self.targets[index])
        return data


def batch_to_device(data, device: torch.device):
    if isinstance(data, dict):
        return {k: v.to(device, non_blocking=True) for k, v in data.items()}
    return data.to(device, non_blocking=True)


class XRVEncoder(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        import torchxrayvision as xrv

        model = xrv.models.DenseNet(weights="densenet121-res224-all" if pretrained else None)
        self.features = model.features
        self.output_dim = 1024

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        return F.adaptive_avg_pool2d(F.relu(x, inplace=False), (1, 1)).flatten(1)


class ImageNetDenseEncoder(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        net = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.features = net.features
        self.output_dim = net.classifier.in_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        return F.adaptive_avg_pool2d(F.relu(x, inplace=False), (1, 1)).flatten(1)


class RadEncoder(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.net = AutoModel.from_pretrained(RAD_DINO_ID)
        else:
            cfg = AutoConfig.from_pretrained(RAD_DINO_ID)
            self.net = AutoModel.from_config(cfg)
        self.output_dim = self.net.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=images).last_hidden_state[:, 0]  # CLS token


class CXRClassifier(nn.Module):
    def __init__(self, model_type: str, pretrained=True, freeze_encoder=True, dropout=0.25, aux_weight=0.0):
        super().__init__()
        self.model_type = model_type
        self.aux_weight = aux_weight
        self.freeze_encoder = freeze_encoder
        if model_type == "fusion":
            if not freeze_encoder:
                raise ValueError("Fusion starter supports frozen backbones only (16GB GPU configuration).")
            self.xrv = XRVEncoder(pretrained)
            self.rad = RadEncoder(pretrained)
            emb_size = self.xrv.output_dim + self.rad.output_dim
        else:
            encoder_types = {
                "imagenet_densenet": ImageNetDenseEncoder,
                "xrv_densenet": XRVEncoder,
                "rad_dino": RadEncoder,
            }
            self.encoder = encoder_types[model_type](pretrained)
            emb_size = self.encoder.output_dim
        if freeze_encoder:
            for p in self.encoder_parameters():
                p.requires_grad_(False)
        self.fusion = nn.Sequential(nn.LayerNorm(emb_size), nn.Dropout(dropout),
                                    nn.Linear(emb_size, 256), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(256, len(CLASSES))
        self.covid_head = nn.Linear(256, 1) if aux_weight > 0 else None

    def encoder_parameters(self):
        if self.model_type == "fusion":
            return list(self.xrv.parameters()) + list(self.rad.parameters())
        return list(self.encoder.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            if self.model_type == "fusion":
                self.xrv.eval()
                self.rad.eval()
            else:
                self.encoder.eval()
        return self

    def forward(self, data):
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            if self.model_type == "fusion":
                emb = torch.cat([self.xrv(data["xrv"]), self.rad(data["rad"])], dim=1)
            else:
                emb = self.encoder(data)
        features = self.fusion(emb)
        return self.classifier(features), (self.covid_head(features).squeeze(1) if self.covid_head is not None else None)


def create_model(config: dict, pretrained=True):
    return CXRClassifier(
        model_type=config["model"], pretrained=pretrained,
        freeze_encoder=config["freeze_encoder"], dropout=config["dropout"],
        aux_weight=config["aux_weight"],
    )


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
