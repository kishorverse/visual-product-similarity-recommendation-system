"""
Central configuration for the Visual Product Similarity project.

Every path and tunable lives here so the pipeline scripts, the notebooks and
the Streamlit app all read from one place. Values can be overridden with
environment variables where it makes sense.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
# Stanford Online Products (Song et al., CVPR 2016) - extracted archive.
SOP_ROOT = Path(os.getenv("SOP_ROOT", DATA_DIR / "raw" / "Stanford_Online_Products"))
SOP_URL = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"

PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_IMAGE_DIR = PROCESSED_DIR / "images"        # resized, validated copies
PRODUCTS_CSV = PROCESSED_DIR / "products.csv"      # one row per product
IMAGES_CSV = PROCESSED_DIR / "images.csv"          # one row per kept image
CLEANING_LOG_CSV = PROCESSED_DIR / "cleaning_log.csv"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "products.db"))

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Dataset selection
# --------------------------------------------------------------------------- #
# The official "test" split (60,502 images / 11,316 products) is the standard
# SOP retrieval benchmark. Use "train" or "all" for other subsets.
SOP_SPLIT = os.getenv("SOP_SPLIT", "test")
# Optional cap for quick experiments (None = keep every product in the split).
MAX_PRODUCTS_PER_CATEGORY = int(os.getenv("MAX_PRODUCTS_PER_CATEGORY", "0")) or None

# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
MIN_IMAGE_SIDE = 64            # images smaller than this are rejected
# Truly blank images (failed uploads) have ~0 pixel std-dev. Product photos on
# white backgrounds can legitimately be low-contrast (std 5-8), so keep this low.
MIN_PIXEL_STD = 2.0
# Cleaned copies are stored with their longest side capped at this size.
# Embedding uses 224px, so this loses nothing and makes every later step faster.
PROCESSED_MAX_SIDE = 320
# Two images whose 64-bit dHashes differ in <= this many bits are candidate
# near-duplicates; a colour pixel check then confirms them (see data_cleaning).
# dHash is greyscale, so without the colour check a red and a blue version of
# the same sofa would be wrongly merged. A pair is confirmed only if *no* pixel
# of the 16x16 RGB thumbnails differs by more than this many levels - colour
# variants change only part of the image, so an average-difference test misses them.
DUPLICATE_HASH_DISTANCE = 4
DUPLICATE_MAX_PIXEL_DIFF = 48
# Image decoding is CPU-bound; beyond ~8 processes the disk becomes the limit
# and each extra Windows worker costs memory.
CLEANING_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 2))

# Simulated commerce metadata (the dataset ships images + labels only).
# A fixed seed keeps prices/stock reproducible between runs.
METADATA_SEED = 42
CATEGORY_PRICE_RANGES = {       # INR, (min, max)
    "bicycle": (6999, 89999),
    "cabinet": (2999, 39999),
    "chair": (999, 24999),
    "coffee_maker": (1499, 29999),
    "fan": (799, 9999),
    "kettle": (499, 5999),
    "lamp": (499, 12999),
    "mug": (149, 1499),
    "sofa": (9999, 149999),
    "stapler": (99, 2499),
    "table": (1999, 49999),
    "toaster": (999, 8999),
}
DEFAULT_PRICE_RANGE = (499, 9999)
OUT_OF_STOCK_RATE = 0.12

# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
# Supported backbones -> embedding dimension after removing the classifier.
MODEL_DIMS = {"resnet50": 2048, "efficientnet_b0": 1280}
# Final pooling over the CNN feature maps, chosen per backbone by the ablation
# in notebook 02 (GeM helps ResNet50 by ~+0.04 product R@1, hurts EfficientNet-B0).
MODEL_POOLING = {"resnet50": "gem", "efficientnet_b0": "avg"}
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "resnet50")
IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
# Average the embedding of the image and its horizontal mirror so a mirrored
# photo of the same product maps to (almost) the same vector.
USE_FLIP_TTA = True
# Mixed precision on GPU roughly doubles throughput; vectors are stored float32.
USE_AMP = True

# --------------------------------------------------------------------------- #
# FAISS index
# --------------------------------------------------------------------------- #
# "hnsw" = approximate nearest neighbour graph (production-like),
# "flat" = exact brute-force inner product (ground truth for ANN recall).
DEFAULT_INDEX_TYPE = os.getenv("INDEX_TYPE", "hnsw")
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# --------------------------------------------------------------------------- #
# Retrieval / evaluation
# --------------------------------------------------------------------------- #
DEFAULT_TOP_K = 10
EVAL_K_VALUES = [1, 5, 10, 20]
# Results are de-duplicated to one image per product, so we fetch more image
# hits than K and keep the best-scoring image of each product.
PRODUCT_OVERFETCH = 8
EVAL_LATENCY_QUERIES = 1000
EVAL_ANN_QUERIES = 5000
EVAL_UPLOAD_QUERIES = 300

# --------------------------------------------------------------------------- #
# Streamlit
# --------------------------------------------------------------------------- #
CATALOG_PAGE_SIZE = 24


def embeddings_path(model_name: str) -> Path:
    """Location of the saved embedding matrix for a backbone."""
    return MODELS_DIR / f"embeddings_{model_name}.npy"


def index_path(model_name: str, index_type: str) -> Path:
    """Location of the saved FAISS index for a backbone / index type."""
    return MODELS_DIR / f"faiss_{model_name}_{index_type}.index"


def ensure_dirs() -> None:
    """Create every output folder the pipeline writes into."""
    for folder in (CLEAN_IMAGE_DIR, MODELS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
