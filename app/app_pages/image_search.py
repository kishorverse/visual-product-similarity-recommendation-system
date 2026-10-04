"""Upload an image (or pick a catalog photo) -> visually similar catalog products."""

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from app.common import (category_label, current_engine, current_filters, results_grid,
                        sort_results, stop_if_not_ready)
from src import database

stop_if_not_ready()

st.title("Search by image")
st.caption("Upload a product photo and get the most visually similar products from "
           "11k+ eBay listings - no text, tags or titles involved.")

engine = current_engine()

source = st.segmented_control("Query source", ["Upload", "Camera", "Catalog photo"],
                              default="Upload", key="search_source")


def pick_random_photo(category: str) -> None:
    rows = engine.catalog.index[engine.catalog["category"] == category].to_numpy()
    st.session_state.sample_row = int(np.random.default_rng().choice(rows))


query_image = None
if source == "Upload":
    upload = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"])
    if upload is not None:
        try:
            query_image = Image.open(upload).convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("That file could not be read as an image.", icon=":material/error:")
elif source == "Camera":
    shot = st.camera_input("Take a photo of a product")
    if shot is not None:
        query_image = Image.open(shot).convert("RGB")
else:
    categories = sorted(engine.catalog["category"].unique())
    with st.container(horizontal=True, vertical_alignment="bottom"):
        category = st.selectbox("Category", categories, format_func=category_label, key="sample_category")
        if st.button("Random photo", icon=":material/shuffle:") or "sample_row" not in st.session_state \
                or engine.catalog.iloc[st.session_state.sample_row]["category"] != category:
            pick_random_photo(category)
    sample = engine.catalog.iloc[st.session_state.sample_row]
    query_image = Image.open(config.CLEAN_IMAGE_DIR / sample["image_file"]).convert("RGB")
    st.caption(f"Query: one photo of **{sample['product_name']}**. Its own listing should come "
               "back at #1 - the other results are visually similar products.")

if query_image is None:
    st.info("Upload an image to start searching.", icon=":material/upload:")
    st.stop()

top_k = st.session_state.get("top_k", config.DEFAULT_TOP_K)
filters = current_filters()

with st.spinner("Extracting embedding and searching..."):
    try:
        outcome = engine.search_image(query_image, top_k=top_k, filters=filters)
    except Exception as exc:  # surface model / index failures without a stack trace
        st.error(f"Search failed: {exc}", icon=":material/error:")
        st.stop()

database.log_search("upload", engine.model_name, top_k, len(outcome.results),
                    outcome.embed_ms + outcome.latency_ms,
                    ",".join(filters.categories) or None)

results = outcome.results
left, right = st.columns([1, 3])
with left:
    with st.container(border=True):
        st.image(query_image, caption="Query image", width="stretch")
with right:
    with st.container(horizontal=True):
        st.metric("CNN embedding", f"{outcome.embed_ms:.0f} ms", border=True)
        st.metric("Search + ranking", f"{outcome.latency_ms:.1f} ms", border=True,
                  help=f"FAISS {engine.index_type.upper()} over {engine.index.ntotal:,} photos, "
                       "grouped to one result per product.")
        st.metric("Products", len(results), border=True)
        if len(results):
            st.metric("Top category", category_label(results["category"].mode()[0]), border=True)
    if len(results):
        st.bar_chart(results["category"].map(category_label).value_counts(), horizontal=True,
                     height=180, x_label="Results", y_label="")

if results.empty:
    st.warning("No products match the current filters.", icon=":material/filter_alt_off:")
    st.stop()

sort_by = st.segmented_control("Sort by", ["Similarity", "Price: low to high",
                                           "Price: high to low", "Rating"],
                               default="Similarity", key="search_sort")
clicked = results_grid(sort_results(results, sort_by or "Similarity"), key_prefix="search")
if clicked:
    st.session_state.selected_product = clicked
    st.switch_page("app_pages/similar_items.py")

with st.expander("Raw results", icon=":material/table_chart:"):
    st.dataframe(results[["rank", "product_id", "product_name", "category", "similarity",
                          "price", "rating", "in_stock", "image_file"]], hide_index=True)
