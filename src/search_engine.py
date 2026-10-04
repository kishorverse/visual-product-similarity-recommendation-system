"""
Steps 3 & 4 - Similarity matching, ranking and filtering.

``VisualSearchEngine`` ties together the CNN, the FAISS index and the SQLite
catalog:

    query image --CNN--> embedding --FAISS--> top image hits --group by product--> top-K products

FAISS holds one vector per *photo*, but a shopper wants *products*: hits are
grouped so each product appears once, represented by its best-matching photo.

Supported queries
  * ``search_image``       - upload / any image -> visually similar products
  * ``similar_to_product`` - "Similar items" for a catalog product. The query is
                             the mean of all the product's photo embeddings
                             (a multi-view product vector), and the product
                             itself is excluded from the results.

Filters (category, price range, in-stock) are pushed *into* the FAISS search
with an ID selector, so K results come back even for selective filters.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import database  # noqa: E402
from src.indexer import load_index  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Optional business filters applied to retrieval."""

    categories: Sequence[str] = field(default_factory=list)
    price_range: Optional[tuple[int, int]] = None
    in_stock_only: bool = False

    def is_empty(self) -> bool:
        return not self.categories and self.price_range is None and not self.in_stock_only


@dataclass
class SearchResult:
    """Ranked product results plus timing information."""

    results: pd.DataFrame
    latency_ms: float          # FAISS search + grouping/ranking time
    embed_ms: float = 0.0      # CNN time (0 for catalog look-ups)


class VisualSearchEngine:
    """Image-based product retrieval over a FAISS index."""

    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL,
        index_type: str = config.DEFAULT_INDEX_TYPE,
        load_extractor: bool = True,
    ):
        self.model_name = model_name
        self.index_type = index_type
        self.index = load_index(config.index_path(model_name, index_type))
        # Memory-mapped: only the rows we touch are read from disk.
        self.embeddings = np.load(config.embeddings_path(model_name), mmap_mode="r")
        self.catalog = database.get_row_mapping(model_name)

        if len(self.catalog) != self.index.ntotal:
            raise RuntimeError(
                f"Catalog has {len(self.catalog)} mapped rows but the index holds "
                f"{self.index.ntotal} vectors - rebuild with `python -m src.pipeline`."
            )
        self._product_ids = self.catalog["product_id"].to_numpy()
        self._rows_by_product = self.catalog.groupby("product_id")["vector_row"].apply(np.array).to_dict()
        self._main_row = self.catalog.sort_values("view_index").groupby("product_id")["vector_row"].first()

        self._extractor = None
        if load_extractor:
            self._load_extractor()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _load_extractor(self):
        from src.feature_extractor import FeatureExtractor  # heavy import, keep lazy
        self._extractor = FeatureExtractor(self.model_name)
        return self._extractor

    @property
    def extractor(self):
        return self._extractor or self._load_extractor()

    @property
    def num_products(self) -> int:
        return len(self._rows_by_product)

    def _allowed_rows(self, filters: Optional[SearchFilters]) -> Optional[np.ndarray]:
        """Vector rows whose product satisfies the filters, or None when unfiltered."""
        if filters is None or filters.is_empty():
            return None
        mask = np.ones(len(self.catalog), dtype=bool)
        if filters.categories:
            mask &= self.catalog["category"].isin(filters.categories).to_numpy()
        if filters.price_range is not None:
            low, high = filters.price_range
            mask &= self.catalog["price"].between(low, high).to_numpy()
        if filters.in_stock_only:
            mask &= self.catalog["in_stock"].astype(bool).to_numpy()
        return self.catalog["vector_row"].to_numpy()[mask].astype("int64")

    def _faiss_search(
        self, query: np.ndarray, k: int, allowed: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run FAISS for ``k`` image hits, restricted to ``allowed`` rows when given."""
        k = min(k, self.index.ntotal if allowed is None else len(allowed))
        if allowed is None:
            return self.index.search(query, k)

        selector = faiss.IDSelectorBatch(allowed)
        if isinstance(self.index, faiss.IndexHNSW):
            params = faiss.SearchParametersHNSW(sel=selector, efSearch=max(config.HNSW_EF_SEARCH, k))
        else:
            params = faiss.SearchParameters(sel=selector)
        scores, rows = self.index.search(query, k, params=params)

        # A graph index can under-fill when the filter is very selective; the
        # eligible subset is then small enough to score exactly.
        if int((rows[0] >= 0).sum()) < k:
            subset_scores = np.asarray(self.embeddings[np.sort(allowed)]) @ query[0]
            order = np.argsort(-subset_scores)[:k]
            scores = subset_scores[order][None, :]
            rows = np.sort(allowed)[order][None, :]
        return scores, rows

    def _rank_products(
        self,
        query: np.ndarray,
        top_k: int,
        filters: Optional[SearchFilters],
        exclude_product: Optional[str] = None,
    ) -> pd.DataFrame:
        """Image hits -> one row per product (best photo), ranked by similarity."""
        query = np.ascontiguousarray(query, dtype="float32")
        allowed = self._allowed_rows(filters)
        if allowed is not None and exclude_product is not None:
            allowed = allowed[self._product_ids[allowed] != exclude_product]
        if allowed is not None and len(allowed) == 0:
            return self._empty()

        fetch = top_k * config.PRODUCT_OVERFETCH
        limit = self.index.ntotal if allowed is None else len(allowed)
        while True:
            scores, rows = self._faiss_search(query, fetch, allowed)
            hits = pd.DataFrame({"vector_row": rows[0], "similarity": scores[0]})
            hits = hits[hits["vector_row"] >= 0]
            hits["product_id"] = self._product_ids[hits["vector_row"].to_numpy()]
            if exclude_product is not None:
                hits = hits[hits["product_id"] != exclude_product]
            best = hits.drop_duplicates("product_id", keep="first")
            # Stop once we have K distinct products or we've searched everything.
            if len(best) >= top_k or fetch >= limit:
                break
            fetch = min(fetch * 4, limit)

        best = best.head(top_k)
        results = self.catalog.iloc[best["vector_row"].to_numpy()].copy()
        results["similarity"] = np.round(best["similarity"].to_numpy(), 4)
        # Rank by similarity; ties broken by rating so better items surface first.
        results = results.sort_values(["similarity", "rating"], ascending=[False, False])
        results.insert(0, "rank", range(1, len(results) + 1))
        return results.reset_index(drop=True)

    def _empty(self) -> pd.DataFrame:
        return self.catalog.iloc[0:0].assign(rank=[], similarity=[])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed(self, image) -> np.ndarray:
        """Embedding for an arbitrary image (path or PIL)."""
        return self.extractor.extract_one(image)

    def product_vector(self, product_id: str) -> np.ndarray:
        """Mean of a product's photo embeddings, re-normalised -> (1, dim)."""
        rows = np.sort(self._rows_by_product[product_id])
        vec = np.asarray(self.embeddings[rows]).mean(axis=0)
        return (vec / np.linalg.norm(vec))[None, :].astype("float32")

    def search_image(
        self,
        image,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[SearchFilters] = None,
    ) -> SearchResult:
        """Top-K catalog products most visually similar to ``image``."""
        t0 = time.perf_counter()
        query = self.embed(image)
        t1 = time.perf_counter()
        results = self._rank_products(query, top_k, filters)
        t2 = time.perf_counter()
        return SearchResult(results, latency_ms=(t2 - t1) * 1000, embed_ms=(t1 - t0) * 1000)

    def similar_to_product(
        self,
        product_id: str,
        top_k: int = config.DEFAULT_TOP_K,
        filters: Optional[SearchFilters] = None,
    ) -> SearchResult:
        """'Similar items' for a catalog product (the product itself is excluded)."""
        if product_id not in self._rows_by_product:
            raise KeyError(f"Unknown product id: {product_id}")
        t0 = time.perf_counter()
        results = self._rank_products(self.product_vector(product_id), top_k, filters,
                                      exclude_product=product_id)
        return SearchResult(results, latency_ms=(time.perf_counter() - t0) * 1000)

    def product(self, product_id: str) -> pd.Series:
        """Catalog row (main photo) for a product id."""
        return self.catalog.iloc[int(self._main_row[product_id])]
