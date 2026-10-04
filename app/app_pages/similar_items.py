"""Browse the catalog page by page and open the 'Similar items' view for any product."""

import math

import streamlit as st

import config
from app.common import (category_label, current_engine, current_filters, product_card, results_grid,
                        stop_if_not_ready)
from src import database

stop_if_not_ready()

st.title("Similar items")
engine = current_engine()
filters = current_filters()
top_k = st.session_state.get("top_k", config.DEFAULT_TOP_K)

# --------------------------------------------------------------------------- #
# Selected product -> its visually similar items
# --------------------------------------------------------------------------- #
selected = st.session_state.get("selected_product")
if selected:
    product = engine.product(selected)
    outcome = engine.similar_to_product(selected, top_k=top_k, filters=filters)
    database.log_search("catalog", engine.model_name, top_k, len(outcome.results),
                        outcome.latency_ms, ",".join(filters.categories) or None)

    photos = database.get_product_images(selected)
    with st.container(border=True):
        left, right = st.columns([1, 3])
        with left:
            st.image(str(config.CLEAN_IMAGE_DIR / product["image_file"]), width="stretch")
        with right:
            st.subheader(product["product_name"])
            stock = ":green-badge[In stock]" if product["in_stock"] else ":red-badge[Out of stock]"
            st.markdown(f"{category_label(product['category'])} · ₹{int(product['price']):,} · "
                        f":material/star: {product['rating']} · {stock} · eBay item `{selected}`")
            st.caption(f"{len(outcome.results)} similar items found in "
                       f"{outcome.latency_ms:.1f} ms ({engine.index_type.upper()} index, "
                       f"{engine.model_name}). The query is the average embedding of all "
                       f"{len(photos)} photos of this listing.")
            gallery = st.columns(min(len(photos), 8))
            for col, photo in zip(gallery, photos[:8]):
                col.image(str(config.CLEAN_IMAGE_DIR / photo), width="stretch")
            if st.button("Close", icon=":material/close:"):
                st.session_state.selected_product = None
                st.rerun()

    if outcome.results.empty:
        st.warning("No similar products match the current filters.",
                   icon=":material/filter_alt_off:")
    else:
        st.markdown("##### Customers who viewed this item also viewed")
        clicked = results_grid(outcome.results, key_prefix="similar")
        if clicked:
            st.session_state.selected_product = clicked
            st.rerun()

# --------------------------------------------------------------------------- #
# Paginated catalog
# --------------------------------------------------------------------------- #
st.markdown("##### Catalog")
page_size = config.CATALOG_PAGE_SIZE
_, total = database.get_catalog_page(filters.categories, filters.price_range,
                                     filters.in_stock_only, limit=1)
num_pages = max(1, math.ceil(total / page_size))

with st.container(horizontal=True, vertical_alignment="center"):
    page_no = st.number_input("Page", 1, num_pages, 1, key="catalog_page", width=140)
    st.caption(f"{total} products · {num_pages} pages · {page_size} per page")

page_df, _ = database.get_catalog_page(filters.categories, filters.price_range,
                                       filters.in_stock_only, limit=page_size,
                                       offset=(int(page_no) - 1) * page_size)
if page_df.empty:
    st.info("No products match the current filters.", icon=":material/filter_alt_off:")
    st.stop()

columns = 6
for start in range(0, len(page_df), columns):
    cols = st.columns(columns)
    for col, (_, row) in zip(cols, page_df.iloc[start:start + columns].iterrows()):
        with col:
            if product_card(row, show_similarity=False, button_key=f"cat_{row['product_id']}"):
                st.session_state.selected_product = row["product_id"]
                st.rerun()
