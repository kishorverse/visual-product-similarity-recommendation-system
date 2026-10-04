"""
Step 2 - Embedding indexing with FAISS.

Two index types are built for each backbone:

  * ``flat``  - IndexFlatIP: exact brute-force inner product. Used as ground
                truth when measuring how much accuracy the ANN index gives up.
  * ``hnsw``  - IndexHNSWFlat: Hierarchical Navigable Small World graph, the
                approximate-nearest-neighbour index used in production. Query
                time grows ~logarithmically with catalog size.

Because the embeddings are L2-normalised, inner product == cosine similarity.

Usage:
    python -m src.indexer --model resnet50
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

logger = logging.getLogger(__name__)

INDEX_TYPES = ("flat", "hnsw")


def build_index(embeddings: np.ndarray, index_type: str = config.DEFAULT_INDEX_TYPE) -> faiss.Index:
    """Build a cosine-similarity FAISS index over L2-normalised ``embeddings``."""
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = config.HNSW_EF_SEARCH
    else:
        raise ValueError(f"Unknown index type '{index_type}'. Choose from {INDEX_TYPES}")

    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """Persist an index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    """Load an index from disk, restoring the HNSW search breadth."""
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}. Run the pipeline first.")
    index = faiss.read_index(str(path))
    if isinstance(index, faiss.IndexHNSW):
        index.hnsw.efSearch = config.HNSW_EF_SEARCH
    return index


def build_and_save_all(model_name: str) -> dict[str, faiss.Index]:
    """Build every index type for a backbone from its saved embeddings."""
    emb_path = config.embeddings_path(model_name)
    if not emb_path.exists():
        raise FileNotFoundError(f"{emb_path} missing - run the feature extractor first.")
    embeddings = np.load(emb_path)

    indexes = {}
    for index_type in INDEX_TYPES:
        start = time.perf_counter()
        index = build_index(embeddings, index_type)
        logger.info("Built %s/%s index with %d vectors in %.3fs",
                    model_name, index_type, index.ntotal, time.perf_counter() - start)
        save_index(index, config.index_path(model_name, index_type))
        indexes[index_type] = index
    return indexes


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build FAISS indexes from saved embeddings")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=list(config.MODEL_DIMS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_and_save_all(args.model)


if __name__ == "__main__":
    main()
