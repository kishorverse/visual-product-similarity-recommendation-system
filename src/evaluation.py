"""
Step 5 - Evaluation.

Protocol (leave-one-out retrieval over the whole catalog)
---------------------------------------------------------
Every catalog photo is used as a query against all *other* photos. SOP gives
two levels of ground truth, so we measure both:

Product level - relevant = another photo of the *same product* (eBay listing)
  Precision@K = same-product photos in top-K / K
  Recall@K    = same-product photos in top-K / all other photos of that product
  Hit@K       = 1 if at least one same-product photo is in the top-K.
                (This is what the SOP / metric-learning literature calls
                "Recall@K".)
  mAP@K       = mean average precision of the top-K list
  Only photos whose product has >= 2 photos can be product-level queries.

Category level - relevant = same category (bicycle, chair, ...)
  Precision@K = same-category photos in top-K / K

Additional checks
-----------------
  * ANN fidelity     - overlap of HNSW top-K with exact (flat) top-K
  * Latency          - single-query FAISS time (flat vs HNSW), CNN time
  * Upload robustness- photos randomly cropped / rotated / colour-shifted to
                       imitate a shopper's own picture: is the right product
                       returned?
  * Visual inspection- query + top-5 grids under reports/figures

Usage:
    python -m src.evaluation
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.indexer import load_index  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metric primitives (pure functions - unit tested)
# --------------------------------------------------------------------------- #
def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """Fraction of the first ``k`` results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    return float(np.sum(relevance[:k])) / k


def recall_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    """Fraction of all relevant catalog items found in the first ``k`` results."""
    if total_relevant <= 0:
        return 0.0
    return float(np.sum(relevance[:k])) / total_relevant


def average_precision_at_k(relevance: Sequence[bool], k: int, total_relevant: int) -> float:
    """AP@K: mean of precision at every rank where a relevant item appears."""
    rel = np.asarray(relevance[:k], dtype=bool)
    if total_relevant <= 0 or not rel.any():
        return 0.0
    hits = np.cumsum(rel)
    precisions = hits[rel] / (np.flatnonzero(rel) + 1)
    return float(precisions.sum() / min(total_relevant, k))


def batch_metrics(relevance: np.ndarray, total_relevant: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """
    Vectorised P@K, R@K, Hit@K and AP@K for many queries at once.

    Args:
        relevance: (N, >=k) boolean matrix, row i = relevance of query i's ranked results
        total_relevant: (N,) number of relevant items in the catalog for each query
    """
    rel = relevance[:, :k].astype(bool)
    hits = rel.sum(axis=1)
    cum = np.cumsum(rel, axis=1)
    ranks = np.arange(1, k + 1)
    ap_num = (cum / ranks * rel).sum(axis=1)
    denom = np.minimum(total_relevant, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(total_relevant > 0, hits / total_relevant, 0.0)
        ap = np.where(denom > 0, ap_num / denom, 0.0)
    return {"P": hits / k, "R": recall, "Hit": (hits > 0).astype(float), "AP": ap}


# --------------------------------------------------------------------------- #
# Leave-one-out retrieval evaluation
# --------------------------------------------------------------------------- #
def leave_one_out_search(
    index: faiss.Index, embeddings: np.ndarray, k: int, query_rows: np.ndarray | None = None,
    batch: int = 8192,
) -> np.ndarray:
    """Top-``k`` neighbours of each query vector, excluding the vector itself."""
    query_rows = np.arange(len(embeddings)) if query_rows is None else np.asarray(query_rows)
    neighbours = np.full((len(query_rows), k), -1, dtype="int64")
    for start in range(0, len(query_rows), batch):
        q = query_rows[start:start + batch]
        _, rows = index.search(np.ascontiguousarray(embeddings[q], dtype="float32"), k + 1)
        for i, (qi, row) in enumerate(zip(q, rows)):
            row = row[(row != qi) & (row >= 0)][:k]
            neighbours[start + i, :len(row)] = row
    return neighbours


def retrieval_metrics(
    neighbours: np.ndarray,
    product_ids: np.ndarray,
    categories: np.ndarray,
    k_values: Sequence[int],
    query_rows: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Product- and category-level metrics at each K.

    Returns:
        (summary dict, per-query DataFrame)
    """
    query_rows = np.arange(len(neighbours)) if query_rows is None else np.asarray(query_rows)
    product_codes = pd.factorize(pd.Series(product_ids))[0]
    images_per_product = np.bincount(product_codes)
    other_views = images_per_product[product_codes[query_rows]] - 1

    valid = neighbours >= 0
    safe = np.where(valid, neighbours, 0)
    same_product = (product_codes[safe] == product_codes[query_rows][:, None]) & valid
    same_category = (categories[safe] == categories[query_rows][:, None]) & valid

    per_query = pd.DataFrame({"query_row": query_rows, "category": categories[query_rows],
                              "other_views": other_views})
    for k in k_values:
        prod = batch_metrics(same_product, other_views, k)
        per_query[f"product_P@{k}"] = prod["P"]
        per_query[f"product_R@{k}"] = prod["R"]
        per_query[f"product_Hit@{k}"] = prod["Hit"]
        per_query[f"product_AP@{k}"] = prod["AP"]
        per_query[f"category_P@{k}"] = same_category[:, :k].sum(axis=1) / k

    # Product metrics only make sense when the product has another photo.
    prod_cols = [c for c in per_query.columns if c.startswith("product_")]
    cat_cols = [c for c in per_query.columns if c.startswith("category_P")]
    has_views = per_query["other_views"] > 0
    overall = {
        **per_query.loc[has_views, prod_cols].mean().round(4).to_dict(),
        **per_query[cat_cols].mean().round(4).to_dict(),
    }
    overall = {k.replace("product_AP@", "product_mAP@"): v for k, v in overall.items()}
    per_category = (
        per_query.loc[has_views].groupby("category")[prod_cols].mean()
        .join(per_query.groupby("category")[cat_cols].mean())
        .round(4).to_dict("index")
    )
    summary = {
        "overall": overall,
        "per_category": per_category,
        "num_queries": int(len(per_query)),
        "num_product_queries": int(has_views.sum()),
    }
    return summary, per_query


def ann_fidelity(flat: faiss.Index, hnsw: faiss.Index, embeddings: np.ndarray,
                 k: int = 10, n_queries: int = config.EVAL_ANN_QUERIES, seed: int = 0) -> float:
    """Average overlap between approximate and exact top-K (1.0 == identical)."""
    rows = np.random.default_rng(seed).choice(len(embeddings), min(n_queries, len(embeddings)), replace=False)
    exact = leave_one_out_search(flat, embeddings, k, rows)
    approx = leave_one_out_search(hnsw, embeddings, k, rows)
    return float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))


def measure_latency(index: faiss.Index, embeddings: np.ndarray, k: int = 10,
                    n_queries: int = config.EVAL_LATENCY_QUERIES, seed: int = 1) -> dict:
    """Single-query FAISS latency in ms (mean and p95) over random catalog vectors."""
    rows = np.random.default_rng(seed).choice(len(embeddings), min(n_queries, len(embeddings)), replace=False)
    queries = np.ascontiguousarray(embeddings[np.sort(rows)], dtype="float32")
    timings = []
    for vec in queries:
        t0 = time.perf_counter()
        index.search(vec[None, :], k)
        timings.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(timings)), 4),
            "p95_ms": round(float(np.percentile(timings, 95)), 4)}


# --------------------------------------------------------------------------- #
# Upload robustness
# --------------------------------------------------------------------------- #
def simulate_user_photo(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Random crop + small rotation + brightness/colour change."""
    w, h = image.size
    scale = rng.uniform(0.75, 0.95)
    cw, ch = int(w * scale), int(h * scale)
    left, top = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    image = image.crop((left, top, left + cw, top + ch))
    image = image.rotate(rng.uniform(-10, 10), resample=Image.BILINEAR, expand=False,
                         fillcolor=(255, 255, 255))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.7, 1.3))
    return image


def upload_robustness(engine, n_queries: int = config.EVAL_UPLOAD_QUERIES,
                      k_values: Sequence[int] = (1, 5, 10), seed: int = 7) -> dict:
    """Does a distorted photo of a catalog item bring back that product?"""
    rng = np.random.default_rng(seed)
    sample = engine.catalog.sample(n=min(n_queries, len(engine.catalog)), random_state=seed)
    max_k = max(k_values)
    ranks, cat_precision, embed_ms, search_ms = [], [], [], []
    for row in sample.itertuples():
        with Image.open(config.CLEAN_IMAGE_DIR / row.image_file) as img:
            query_img = simulate_user_photo(img.convert("RGB"), rng)
        out = engine.search_image(query_img, top_k=max_k)
        embed_ms.append(out.embed_ms)
        search_ms.append(out.latency_ms)
        hit = np.flatnonzero(out.results["product_id"].to_numpy() == row.product_id)
        ranks.append(int(hit[0]) + 1 if hit.size else None)
        cat_precision.append(float(np.mean(out.results["category"].to_numpy()[:10] == row.category)))

    result = {f"exact_product_hit@{k}": round(float(np.mean([r is not None and r <= k for r in ranks])), 4)
              for k in k_values}
    result["category_P@10"] = round(float(np.mean(cat_precision)), 4)
    result["n_queries"] = len(sample)
    result["cnn_embed_ms_per_image"] = round(float(np.median(embed_ms)), 2)
    result["search_and_rank_ms"] = round(float(np.median(search_ms)), 2)
    return result


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_model_comparison(results: dict, path: Path) -> None:
    """Grouped bars: product Hit@K and category Precision@K per backbone."""
    k_values = config.EVAL_K_VALUES
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.38
    x = np.arange(len(k_values))
    panels = (("product_Hit@", "Product Recall@K (Hit@K, SOP benchmark)"),
              ("category_P@", "Category Precision@K"))
    for i, (model, res) in enumerate(results.items()):
        overall = res["retrieval"]["overall"]
        for ax, (prefix, _) in zip(axes, panels):
            values = [overall[f"{prefix}{k}"] for k in k_values]
            bars = ax.bar(x + (i - 0.5) * width, values, width, label=model)
            ax.bar_label(bars, fmt="%.2f", fontsize=8)
    for ax, (_, title) in zip(axes, panels):
        ax.set_xticks(x, [f"K={k}" for k in k_values])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_category(per_category: dict, model: str, path: Path) -> None:
    """Product Hit@1 and category P@10 for each category."""
    df = pd.DataFrame(per_category).T.sort_values("product_Hit@1")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    b1 = ax.barh(y + 0.2, df["product_Hit@1"], 0.4, label="Product Recall@1", color="#4C72B0")
    b2 = ax.barh(y - 0.2, df["category_P@10"], 0.4, label="Category Precision@10", color="#DD8452")
    ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
    ax.set_yticks(y, df.index)
    ax.set_xlim(0, 1.1)
    ax.set_title(f"Per-category retrieval quality ({model})")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_visual_inspection(
    neighbours: np.ndarray, query_rows: np.ndarray, embeddings: np.ndarray, catalog: pd.DataFrame,
    path: Path, top_n: int = 5, seed: int = 3,
) -> None:
    """One query per category: query photo + top-N photos (green same product, blue same category)."""
    rng = np.random.default_rng(seed)
    position = {int(r): i for i, r in enumerate(query_rows)}
    eligible = catalog[catalog["vector_row"].isin(query_rows)]
    picks = (eligible.groupby("category")["vector_row"]
             .apply(lambda s: int(rng.choice(s.to_numpy()))).to_numpy())
    picks = rng.choice(picks, size=min(8, len(picks)), replace=False)

    fig, axes = plt.subplots(len(picks), top_n + 1, figsize=(2.1 * (top_n + 1), 2.15 * len(picks)))
    for r, q in enumerate(picks):
        q_info = catalog.iloc[q]
        for c, item in enumerate([q, *neighbours[position[q]][:top_n]]):
            ax = axes[r, c]
            info = catalog.iloc[item]
            with Image.open(config.CLEAN_IMAGE_DIR / info["image_file"]) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f"QUERY\n{info['category']}", fontsize=8, fontweight="bold")
                colour = "black"
            else:
                sim = float(np.dot(embeddings[q], embeddings[item]))
                if info["product_id"] == q_info["product_id"]:
                    colour, tag = "#2ca02c", "same product"
                elif info["category"] == q_info["category"]:
                    colour, tag = "#1f77b4", info["category"]
                else:
                    colour, tag = "#d62728", info["category"]
                ax.set_title(f"#{c} {tag}\ncos={sim:.3f}", fontsize=7.5, color=colour)
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_model(model_name: str, run_upload_test: bool = True) -> dict:
    """Run every evaluation for one backbone and return a JSON-able dict."""
    from src import database

    embeddings = np.load(config.embeddings_path(model_name))
    catalog = database.get_row_mapping(model_name)
    product_ids = catalog["product_id"].to_numpy()
    categories = catalog["category"].to_numpy()
    flat = load_index(config.index_path(model_name, "flat"))
    hnsw = load_index(config.index_path(model_name, "hnsw"))

    query_rows = np.arange(len(embeddings))
    t0 = time.perf_counter()
    neighbours = leave_one_out_search(hnsw, embeddings, max(config.EVAL_K_VALUES), query_rows)
    logger.info("  leave-one-out search over %d queries: %.1fs", len(query_rows), time.perf_counter() - t0)
    summary, per_query = retrieval_metrics(neighbours, product_ids, categories,
                                           config.EVAL_K_VALUES, query_rows)
    per_query.to_csv(config.REPORTS_DIR / f"per_query_metrics_{model_name}.csv", index=False)

    result = {
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_images": int(len(embeddings)),
        "num_products": int(catalog["product_id"].nunique()),
        "num_categories": int(catalog["category"].nunique()),
        "retrieval": summary,
        "ann_fidelity@10": round(ann_fidelity(flat, hnsw, embeddings, 10), 4),
        "latency": {"flat": measure_latency(flat, embeddings), "hnsw": measure_latency(hnsw, embeddings)},
    }
    del flat

    plot_per_category(summary["per_category"], model_name,
                      config.FIGURES_DIR / f"per_category_{model_name}.png")
    plot_visual_inspection(neighbours, query_rows, embeddings, catalog,
                           config.FIGURES_DIR / f"visual_inspection_{model_name}.png")

    if run_upload_test:
        from src.search_engine import VisualSearchEngine
        engine = VisualSearchEngine(model_name, "hnsw")
        result["upload_robustness"] = upload_robustness(engine)
    return result


def run_evaluation(models: Sequence[str] = tuple(config.MODEL_DIMS)) -> dict:
    """Evaluate every backbone that has been indexed, save metrics + figures."""
    config.ensure_dirs()
    results = {}
    for model in models:
        if not config.index_path(model, "hnsw").exists():
            logger.warning("Skipping %s - no index built", model)
            continue
        logger.info("Evaluating %s ...", model)
        results[model] = evaluate_model(model)
        overall = results[model]["retrieval"]["overall"]
        logger.info("%s: product Hit@1=%.4f  Hit@10=%.4f  category P@10=%.4f", model,
                    overall["product_Hit@1"], overall["product_Hit@10"], overall["category_P@10"])

    if not results:
        raise RuntimeError("No indexed models found - run the pipeline first.")
    if len(results) > 1:
        plot_model_comparison(results, config.FIGURES_DIR / "model_comparison.png")
    best = max(results, key=lambda m: results[m]["retrieval"]["overall"]["product_Hit@1"])
    payload = {"best_model": best, "selection_metric": "product_Hit@1", "models": results}
    config.METRICS_JSON.write_text(json.dumps(payload, indent=2))
    logger.info("Best model by product Recall@1: %s -> %s", best, config.METRICS_JSON)
    return payload


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_evaluation()


if __name__ == "__main__":
    main()
