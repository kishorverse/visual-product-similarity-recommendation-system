"""
Step 0 - Data cleaning for Stanford Online Products (SOP).

SOP layout (after extracting the archive):

    Stanford_Online_Products/
        Ebay_test.txt / Ebay_train.txt / Ebay_info.txt   "image_id class_id super_class_id path"
        bicycle_final/251952414262_0.JPG                  <eBay item id>_<view index>.JPG
        ...

Produces:
  * data/processed/images/<category>/<file>.jpg   validated, RGB, longest side <= 320px
  * data/processed/products.csv                   one row per product (+ simulated price/stock)
  * data/processed/images.csv                     one row per kept image
  * data/processed/cleaning_log.csv               what happened to every source image and why

Checks applied to every image:
  1. file exists and fully decodes (corrupt / truncated files rejected)
  2. minimum resolution
  3. not a near-uniform (blank) image
  4. greyscale / palette / CMYK converted to RGB
  5. near-duplicate removal: 64-bit dHash candidate match, confirmed on pixels

Usage:
    python -m src.data_cleaning
"""

import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

logger = logging.getLogger(__name__)

THUMB_SIDE = 16   # colour thumbnail used to confirm duplicate candidates
HASH_CHUNKS = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))


# --------------------------------------------------------------------------- #
# Loading the SOP index
# --------------------------------------------------------------------------- #
def load_sop_index(root: Path = config.SOP_ROOT, split: str = config.SOP_SPLIT) -> pd.DataFrame:
    """
    Read the official SOP split file into a tidy DataFrame.

    Columns: image_id, class_id, category, product_id (eBay item id), view, source_path
    """
    split_files = {"test": "Ebay_test.txt", "train": "Ebay_train.txt", "all": "Ebay_info.txt"}
    if split not in split_files:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(split_files)}")
    index_file = Path(root) / split_files[split]
    if not index_file.exists():
        raise FileNotFoundError(
            f"{index_file} not found. Download the dataset first: "
            "python scripts/download_dataset.py"
        )

    df = pd.read_csv(index_file, sep=" ")
    folder = df["path"].str.split("/").str[0]
    stem = df["path"].str.split("/").str[1].str.rsplit(".", n=1).str[0]
    return pd.DataFrame({
        "image_id": df["image_id"].astype(int),
        "class_id": df["class_id"].astype(int),
        "category": folder.str.replace("_final", "", regex=False),
        "product_id": stem.str.rsplit("_", n=1).str[0],
        "view": stem.str.rsplit("_", n=1).str[1].astype(int),
        "source_path": df["path"],
    })


def sample_products(index: pd.DataFrame, max_per_category: int | None, seed: int = 0) -> pd.DataFrame:
    """Optionally keep only N random products per category (all their images)."""
    if not max_per_category:
        return index
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in index.groupby("category"):
        products = group["product_id"].unique()
        keep.extend(rng.choice(products, size=min(max_per_category, len(products)), replace=False))
    return index[index["product_id"].isin(set(keep))].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #
def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    """
    64-bit dHash: shrink to 9x8 greyscale and compare neighbouring pixels.

    Robust to re-encoding, resizing and small colour shifts - exactly what a
    re-used listing photo looks like.
    """
    grey = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def colour_thumbnail(image: Image.Image) -> np.ndarray:
    """Small RGB thumbnail used to confirm a hash match on actual (colour) pixels."""
    return np.asarray(image.convert("RGB").resize((THUMB_SIDE, THUMB_SIDE), Image.BILINEAR), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Per-image processing (runs in worker processes)
# --------------------------------------------------------------------------- #
def process_image(task: tuple[str, str]) -> dict:
    """
    Validate, convert and resize one image.

    Args:
        task: (absolute source path, absolute destination path)
    """
    src, dst = task
    out = {"status": "rejected", "reason": "", "orig_width": 0, "orig_height": 0,
           "width": 0, "height": 0, "mode": "", "dhash": 0, "thumb": None}
    try:
        with Image.open(src) as img:
            img.load()
            out["mode"] = img.mode
            out["orig_width"], out["orig_height"] = img.size
            image = img.convert("RGB")
    except FileNotFoundError:
        out["reason"] = "missing file"
        return out
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        out["reason"] = f"corrupt/unreadable ({type(exc).__name__})"
        return out

    if min(image.size) < config.MIN_IMAGE_SIDE:
        out["reason"] = f"too small ({image.size[0]}x{image.size[1]})"
        return out
    if float(np.asarray(image.convert("L")).std()) < config.MIN_PIXEL_STD:
        out["reason"] = "near-uniform / blank image"
        return out

    image.thumbnail((config.PROCESSED_MAX_SIDE, config.PROCESSED_MAX_SIDE), Image.LANCZOS)
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, "JPEG", quality=92)
    except OSError as exc:
        out["reason"] = f"failed to save ({exc})"
        return out

    out.update(status="kept", width=image.size[0], height=image.size[1],
               dhash=difference_hash(image), thumb=colour_thumbnail(image))
    return out


# --------------------------------------------------------------------------- #
# Scalable near-duplicate detection
# --------------------------------------------------------------------------- #
def find_duplicates(
    hashes: np.ndarray,
    thumbs: np.ndarray,
    max_bits: int = config.DUPLICATE_HASH_DISTANCE,
    max_pixel_diff: int = config.DUPLICATE_MAX_PIXEL_DIFF,
) -> dict[int, int]:
    """
    Return {duplicate_index: original_index} for near-duplicate images.

    Comparing every pair of 60k images is 1.8 billion comparisons, so we use
    the pigeonhole principle: split each 64-bit hash into 5 chunks. If two
    hashes differ in <= 4 bits, at least one chunk is identical, so only
    images that share a chunk value need to be compared.

    A dHash match alone is not enough on this dataset. dHash is greyscale, so
    colour variants of one design (a red and a blue stapler) collide, and
    simple products on plain white backgrounds have similar coarse gradients.
    Candidates are therefore confirmed on a 16x16 RGB thumbnail: every pixel
    must be within ``max_pixel_diff`` levels on every channel. (A mean-difference
    test is not enough - a colour variant changes only part of the frame.)
    """
    hashes = hashes.astype(np.uint64)
    candidate_pairs: set[tuple[int, int]] = set()

    for lo, hi in HASH_CHUNKS:
        mask = np.uint64((1 << (hi - lo)) - 1)
        chunk = (hashes >> np.uint64(lo)) & mask
        order = np.argsort(chunk, kind="stable")
        sorted_chunk = chunk[order]
        boundaries = np.flatnonzero(np.diff(sorted_chunk)) + 1
        for group in np.split(order, boundaries):
            if len(group) < 2:
                continue
            group = np.sort(group)
            # Compare the group in blocks to bound memory on large buckets.
            for start in range(0, len(group), 2048):
                block = group[start:start + 2048]
                dist = np.bitwise_count(hashes[block][:, None] ^ hashes[group][None, :])
                rows, cols = np.nonzero(dist <= max_bits)
                for a, b in zip(block[rows], group[cols]):
                    if a < b:
                        candidate_pairs.add((int(a), int(b)))

    duplicates: dict[int, int] = {}
    for a, b in sorted(candidate_pairs, key=lambda p: (p[1], p[0])):
        if b in duplicates or a in duplicates:
            continue  # b already removed, or a is itself a removed copy
        diff = np.abs(thumbs[a].astype(np.int16) - thumbs[b].astype(np.int16)).max()
        if diff <= max_pixel_diff:
            duplicates[b] = a
    return duplicates


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def attach_simulated_metadata(products: pd.DataFrame) -> pd.DataFrame:
    """
    Add price / rating / stock columns per product.

    SOP provides images and labels only, but the brief asks for optional
    filtering by price and availability. Values are simulated with a fixed
    seed and category-specific price ranges so results are reproducible.
    """
    rng = np.random.default_rng(config.METADATA_SEED)
    prices, ratings = [], []
    for category in products["category"]:
        low, high = config.CATEGORY_PRICE_RANGES.get(category, config.DEFAULT_PRICE_RANGE)
        price = np.exp(rng.uniform(np.log(low), np.log(high)))   # cheap items more common
        prices.append(int(round(price / 10.0) * 10 - 1))          # e.g. 1299
        ratings.append(round(float(np.clip(rng.normal(4.1, 0.45), 1.0, 5.0)), 1))
    products["price"] = prices
    products["rating"] = ratings
    products["in_stock"] = rng.random(len(products)) > config.OUT_OF_STOCK_RATE
    return products


# --------------------------------------------------------------------------- #
# Main cleaning routine
# --------------------------------------------------------------------------- #
def clean_dataset(
    root: Path = config.SOP_ROOT,
    split: str = config.SOP_SPLIT,
    max_products_per_category: int | None = config.MAX_PRODUCTS_PER_CATEGORY,
    out_dir: Path = config.CLEAN_IMAGE_DIR,
    workers: int = config.CLEANING_WORKERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean the selected SOP split.

    Returns:
        (products, images, cleaning_log) DataFrames - also written to CSV.
    """
    index = sample_products(load_sop_index(root, split), max_products_per_category)
    index = index.sort_values("image_id").reset_index(drop=True)
    index["file_name"] = index["category"] + "/" + index["source_path"].str.split("/").str[1] \
        .str.rsplit(".", n=1).str[0] + ".jpg"
    logger.info("Cleaning %d images of %d products (%s split) with %d workers",
                len(index), index["product_id"].nunique(), split, workers)

    tasks = [(str(Path(root) / s), str(Path(out_dir) / f))
             for s, f in zip(index["source_path"], index["file_name"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_image, tasks, chunksize=256))

    log = index.copy()
    for key in ("status", "reason", "orig_width", "orig_height", "width", "height", "mode"):
        log[key] = [r[key] for r in results]
    log["dhash"] = [f"{r['dhash']:016x}" if r["status"] == "kept" else "" for r in results]

    # Near-duplicates among the images that passed every other check.
    kept_idx = np.flatnonzero(log["status"].to_numpy() == "kept")
    hashes = np.array([results[i]["dhash"] for i in kept_idx], dtype=np.uint64)
    thumbs = np.stack([results[i]["thumb"] for i in kept_idx])
    duplicates = find_duplicates(hashes, thumbs)
    for dup, orig in duplicates.items():
        d, o = kept_idx[dup], kept_idx[orig]
        same = log.at[d, "product_id"] == log.at[o, "product_id"]
        log.at[d, "status"] = "rejected"
        log.at[d, "reason"] = (f"near-duplicate ({'same product' if same else 'other product'}) "
                               f"of {log.at[o, 'file_name']}")
        (Path(out_dir) / log.at[d, "file_name"]).unlink(missing_ok=True)

    images = log[log["status"] == "kept"][[
        "image_id", "product_id", "class_id", "category", "view", "file_name",
        "orig_width", "orig_height", "width", "height", "mode", "dhash",
    ]].reset_index(drop=True)

    products = (images.groupby("product_id", sort=False)
                .agg(class_id=("class_id", "first"), category=("category", "first"),
                     num_images=("image_id", "size"), main_image_id=("image_id", "min"))
                .reset_index()
                .sort_values(["category", "class_id"])
                .reset_index(drop=True))
    products["product_name"] = (
        products["category"].str.replace("_", " ").str.title()
        + " #" + (products.groupby("category").cumcount() + 1).astype(str).str.zfill(4)
    )
    products = attach_simulated_metadata(products)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    products.to_csv(config.PRODUCTS_CSV, index=False)
    images.to_csv(config.IMAGES_CSV, index=False)
    log.drop(columns=["source_path"]).to_csv(config.CLEANING_LOG_CSV, index=False)

    logger.info("Kept %d / %d images, %d products, %d categories; rejected %d",
                len(images), len(log), len(products), products["category"].nunique(),
                int((log["status"] == "rejected").sum()))
    return products, images, log


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config.ensure_dirs()
    products, images, log = clean_dataset()
    print(images["category"].value_counts().to_string())
    reasons = log.loc[log["status"] == "rejected", "reason"].str.replace(r" of .*", "", regex=True)
    print("\nRejected:\n" + reasons.value_counts().to_string())


if __name__ == "__main__":
    main()
