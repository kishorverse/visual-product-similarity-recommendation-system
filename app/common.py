"""Shared resources and widgets for the Streamlit pages."""

import json

import pandas as pd
import streamlit as st

import config
from src import database
from src.search_engine import SearchFilters, VisualSearchEngine

MODEL_LABELS = {"resnet50": "ResNet50 (2048-d)", "efficientnet_b0": "EfficientNet-B0 (1280-d)"}
INDEX_LABELS = {"hnsw": "HNSW (approximate)", "flat": "Flat (exact)"}


def artifacts_ready() -> bool:
    """True when the pipeline has produced the DB and at least one index."""
    return config.DB_PATH.exists() and any(
        config.index_path(m, "hnsw").exists() for m in config.MODEL_DIMS
    )


def available_models() -> list[str]:
    return [m for m in config.MODEL_DIMS if config.index_path(m, "hnsw").exists()]


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not config.METRICS_JSON.exists():
        return {}
    return json.loads(config.METRICS_JSON.read_text())


@st.cache_resource(show_spinner="Loading CNN and FAISS index...")
def get_engine(model_name: str, index_type: str) -> VisualSearchEngine:
    """One engine per (model, index) for the whole server - the CNN is expensive to load."""
    return VisualSearchEngine(model_name, index_type)


@st.cache_data(show_spinner=False)
def get_filter_options() -> tuple[list[str], tuple[int, int]]:
    return database.get_categories(), database.get_price_bounds()


def stop_if_not_ready() -> None:
    if not artifacts_ready():
        st.error(
            "No index found. Build it first from the project root:\n\n"
            "```bash\npython -m src.pipeline\n```",
            icon=":material/error:",
        )
        st.stop()


def render_sidebar() -> None:
    """Model / retrieval settings and business filters, shared by every page."""
    if not artifacts_ready():
        return
    metrics = load_metrics()
    models = available_models()
    best = metrics.get("best_model", models[0])

    with st.sidebar:
        st.header("Retrieval settings")
        st.selectbox("Embedding model", models, index=models.index(best) if best in models else 0,
                     format_func=lambda m: MODEL_LABELS.get(m, m), key="model_name")
        st.segmented_control("FAISS index", list(INDEX_LABELS), default="hnsw",
                             format_func=lambda i: INDEX_LABELS[i].split(" ")[0],
                             key="index_type", help="HNSW is approximate and scales to millions "
                             "of items; Flat is exact brute force.")
        st.slider("Top-K results", 1, 30, config.DEFAULT_TOP_K, key="top_k")

        st.header("Filters")
        categories, (low, high) = get_filter_options()
        st.multiselect("Category", categories, key="filter_categories",
                       format_func=category_label,
                       placeholder="All categories")
        st.slider("Price (₹)", low, high, (low, high), step=100, key="filter_price")
        st.toggle("In stock only", key="filter_in_stock")
        st.caption("Images: Stanford Online Products (eBay listings). Price, rating "
                   "and stock are simulated - the dataset ships images and labels only.")


def current_filters() -> SearchFilters:
    """Translate sidebar widgets into a ``SearchFilters`` object."""
    _, bounds = get_filter_options()
    price = st.session_state.get("filter_price", bounds)
    return SearchFilters(
        categories=st.session_state.get("filter_categories", []),
        price_range=None if tuple(price) == tuple(bounds) else tuple(price),
        in_stock_only=st.session_state.get("filter_in_stock", False),
    )


def current_engine() -> VisualSearchEngine:
    return get_engine(st.session_state.get("model_name", available_models()[0]),
                      st.session_state.get("index_type") or "hnsw")


def sort_results(results: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Re-order ranked results by a user-chosen key (rank is kept for reference)."""
    if sort_by == "Price: low to high":
        return results.sort_values("price")
    if sort_by == "Price: high to low":
        return results.sort_values("price", ascending=False)
    if sort_by == "Rating":
        return results.sort_values("rating", ascending=False)
    return results


def category_label(category: str) -> str:
    return category.replace("_", " ").title()


def product_card(row: pd.Series, show_similarity: bool = True, button_key: str | None = None) -> bool:
    """Render one product tile. Returns True if its 'Find similar' button was clicked."""
    with st.container(border=True):
        st.image(str(config.CLEAN_IMAGE_DIR / row["image_file"]), width="stretch")
        st.markdown(f"**{row['product_name']}**")
        stock = ":green-badge[In stock]" if row["in_stock"] else ":red-badge[Out of stock]"
        st.markdown(f"₹{int(row['price']):,} · :material/star: {row['rating']} · {stock}")
        if "num_images" in row:
            st.caption(f"{category_label(row['category'])} · {int(row['num_images'])} photos")
        if show_similarity and "similarity" in row:
            st.progress(max(0.0, min(1.0, float(row["similarity"]))),
                        text=f"#{int(row['rank'])} · similarity {row['similarity']:.3f}")
        if button_key:
            return st.button("Find similar", key=button_key, icon=":material/search:",
                             width="stretch")
    return False


def results_grid(results: pd.DataFrame, columns: int = 5, key_prefix: str = "res") -> str | None:
    """Grid of product cards; returns the product id whose button was clicked, if any."""
    clicked = None
    for start in range(0, len(results), columns):
        cols = st.columns(columns)
        for col, (_, row) in zip(cols, results.iloc[start:start + columns].iterrows()):
            with col:
                if product_card(row, button_key=f"{key_prefix}_{row['product_id']}"):
                    clicked = row["product_id"]
    return clicked
