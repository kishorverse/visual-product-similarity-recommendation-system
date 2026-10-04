"""
Step 1 - Image feature extraction.

A pretrained ImageNet CNN (ResNet50 or EfficientNet-B0) with its final
classification layer removed turns every product image into a dense vector:

    image -> backbone -> GeM pool (p=3)         -> [2048] (ResNet50)
    image -> backbone -> global average pool    -> [1280] (EfficientNet-B0)

The pooling choice per backbone comes from the ablation in notebook 02:
GeM (generalised-mean) pooling lifts ResNet50's product Recall@1 by ~0.04
but slightly hurts EfficientNet-B0, so each backbone uses what works for it.

Vectors are L2-normalised, so the inner product of two embeddings *is* their
cosine similarity. That lets FAISS do cosine search with an inner-product
index.

Usage:
    python -m src.feature_extractor --model resnet50
    python -m src.feature_extractor --model efficientnet_b0
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image]


def build_transform(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """
    ImageNet preprocessing.

    We resize the whole image to a square instead of center-cropping: product
    photos often have the item near the edge, and cropping would cut it off.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class GeM(nn.Module):
    """
    Generalised-mean pooling: (mean(x^p))^(1/p) over each feature map.

    p=1 is average pooling and p -> inf is max pooling; p=3 emphasises the
    strongest activations (the object) over background, a standard choice
    for image retrieval. Computed in float32 because x^3 can overflow fp16.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p, self.eps = p, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().clamp(min=self.eps).pow(self.p)
            return nn.functional.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)


def build_backbone(model_name: str) -> nn.Module:
    """Load a pretrained backbone and strip its classifier head."""
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if config.MODEL_POOLING[model_name] == "gem":
            model.avgpool = GeM()
        model.fc = nn.Identity()
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if config.MODEL_POOLING[model_name] == "gem":
            model.avgpool = GeM()
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(config.MODEL_DIMS)}")
    return model.eval()


class ImageFolderDataset(Dataset):
    """Minimal dataset over an explicit list of image paths."""

    def __init__(self, paths: Sequence[Path], transform: transforms.Compose):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


class FeatureExtractor:
    """Wraps a headless CNN and turns images into L2-normalised embeddings."""

    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL,
        device: str | None = None,
        use_flip_tta: bool = config.USE_FLIP_TTA,
    ):
        self.model_name = model_name
        self.dim = config.MODEL_DIMS[model_name]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_flip_tta = use_flip_tta
        self.transform = build_transform()
        self.model = build_backbone(model_name).to(self.device)
        # One dummy pass so CUDA/cuDNN initialisation is not charged to the first real query.
        self._embed_batch(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
        logger.info("Loaded %s on %s (dim=%d, flip_tta=%s)",
                    model_name, self.device, self.dim, use_flip_tta)

    @torch.inference_mode()
    def _embed_batch(self, batch: torch.Tensor, amp: bool = False) -> np.ndarray:
        batch = batch.to(self.device, non_blocking=True)
        use_amp = amp and config.USE_AMP and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            features = self.model(batch)
            if self.use_flip_tta:
                features = features + self.model(torch.flip(batch, dims=[3]))
        # Normalise in float32 so unit length is exact regardless of AMP.
        features = nn.functional.normalize(features.float(), p=2, dim=1)
        return features.cpu().numpy()

    def extract_paths(
        self,
        paths: Sequence[Path],
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
    ) -> np.ndarray:
        """
        Embed many image files in batches -> (N, dim) float32 matrix.

        JPEG decoding is the bottleneck, so ``num_workers`` background processes
        decode the next batches while the GPU embeds the current one.
        """
        loader = DataLoader(
            ImageFolderDataset(paths, self.transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )
        out = np.empty((len(paths), self.dim), dtype="float32")
        log_every = max(1, len(loader) // 10)
        start = 0
        for i, batch in enumerate(loader, start=1):
            vectors = self._embed_batch(batch, amp=True)
            out[start:start + len(vectors)] = vectors
            start += len(vectors)
            if i % log_every == 0 or i == len(loader):
                logger.info("  embedded %d / %d images", start, len(paths))
        return out

    def extract_one(self, image: ImageInput) -> np.ndarray:
        """Embed a single image (path or PIL image) -> (1, dim) float32 matrix."""
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                image = img.convert("RGB")
        else:
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        # Single queries run in float32: the fp16 autocast path pays a ~3 s setup
        # cost the first time it runs in each new thread, and Streamlit serves
        # every rerun from a fresh thread. One image doesn't benefit from AMP anyway.
        return self._embed_batch(tensor).astype("float32")


def extract_catalog_embeddings(model_name: str, catalog_paths: Sequence[Path]) -> np.ndarray:
    """Embed the whole catalog with ``model_name`` and save the matrix to disk."""
    extractor = FeatureExtractor(model_name)
    start = time.perf_counter()
    embeddings = extractor.extract_paths(catalog_paths)
    elapsed = time.perf_counter() - start
    logger.info("Extracted %s embeddings %s in %.1fs (%.1f img/s)",
                model_name, embeddings.shape, elapsed, len(catalog_paths) / elapsed)

    out_path = config.embeddings_path(model_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    return embeddings


def main() -> None:
    """CLI entry point."""
    import pandas as pd
    from src import database

    parser = argparse.ArgumentParser(description="Extract CNN embeddings for the catalog")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_DIMS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    images = pd.read_csv(config.IMAGES_CSV, dtype={"product_id": str})
    paths = [config.CLEAN_IMAGE_DIR / f for f in images["file_name"]]
    embeddings = extract_catalog_embeddings(args.model, paths)
    database.register_embeddings(args.model, embeddings.shape[1], images["image_id"].tolist())


if __name__ == "__main__":
    main()
