"""Pipeline overview, dataset summary and search usage."""

import pandas as pd
import streamlit as st

import config
from app.common import stop_if_not_ready
from src import database

stop_if_not_ready()

st.title("How it works")

st.markdown(
    """
**Data** - [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
(Song et al., CVPR 2016): eBay listings, each photographed from several angles. We use
the official test split - 12 categories, 11k+ products, ~59k photos after cleaning.

**1. Clean** - validate every image, drop corrupt / blank / tiny files, convert to RGB,
resize to 320px, and remove re-used photos with a perceptual hash confirmed on colour
pixels (so a red and a blue version of the same chair are *not* merged).

**2. Embed** - a pretrained CNN (ResNet50 or EfficientNet-B0) with its classifier
removed turns each photo into a 2048-d / 1280-d vector, L2-normalised so that the
dot product equals cosine similarity.

**3. Index** - vectors go into a FAISS HNSW graph (approximate nearest neighbour)
plus an exact flat index used as ground truth.

**4. Search & rank** - a query image is embedded the same way, FAISS returns the
closest photos, hits are grouped to one result per product (its best photo), and
SQLite supplies the product metadata. Category, price and stock filters are pushed
into the FAISS search itself.

**5. Evaluate** - product- and category-level Precision@K / Recall@K, mAP, latency,
ANN fidelity and visual inspection.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Dataset after cleaning")
    if config.CLEANING_LOG_CSV.exists():
        log = pd.read_csv(config.CLEANING_LOG_CSV)
        summary = (log.assign(outcome=log["reason"].fillna("kept").str.split(" of ").str[0]
                              .str.split(" \\(").str[0])
                   .groupby("outcome").size().rename("images").reset_index())
        with st.container(horizontal=True):
            st.metric("Source photos", f"{len(log):,}", border=True)
            st.metric("Kept", f"{int((log['status'] == 'kept').sum()):,}", border=True)
            st.metric("Removed", f"{int((log['status'] == 'rejected').sum()):,}", border=True)
        st.dataframe(summary, hide_index=True)
with col2:
    st.markdown("##### Search usage (from SQLite `search_logs`)")
    stats = database.get_search_stats()
    if stats.empty:
        st.caption("No searches logged yet.")
    else:
        st.dataframe(stats, hide_index=True)
