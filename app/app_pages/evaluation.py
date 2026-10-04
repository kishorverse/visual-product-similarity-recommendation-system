"""Offline evaluation: product & category Precision@K / Recall@K, latency, visual inspection."""

import pandas as pd
import streamlit as st

import config
from app.common import MODEL_LABELS, category_label, load_metrics, stop_if_not_ready

stop_if_not_ready()

st.title("Evaluation")
metrics = load_metrics()
if not metrics:
    st.warning("No metrics yet - run `python -m src.evaluation`.", icon=":material/warning:")
    st.stop()

models = metrics["models"]
st.caption(
    "Leave-one-out over every catalog photo: each photo queries all the others. "
    "**Product level** - relevant = another photo of the *same eBay listing* "
    "(Recall@K here is the hit rate, the standard Stanford Online Products benchmark metric). "
    "**Category level** - relevant = same category. "
    f"Best model: **{MODEL_LABELS.get(metrics['best_model'], metrics['best_model'])}**."
)

model = st.segmented_control("Model", list(models), default=metrics["best_model"],
                             format_func=lambda m: MODEL_LABELS.get(m, m), key="eval_model")
res = models[model or metrics["best_model"]]
overall = res["retrieval"]["overall"]

with st.container(horizontal=True):
    st.metric("Product Recall@1", f"{overall['product_Hit@1']:.3f}", border=True,
              help="Share of queries whose #1 result is another photo of the same product.")
    st.metric("Product Recall@10", f"{overall['product_Hit@10']:.3f}", border=True)
    st.metric("Category Precision@10", f"{overall['category_P@10']:.3f}", border=True)
    st.metric("HNSW search", f"{res['latency']['hnsw']['mean_ms']:.2f} ms", border=True,
              help=f"Exact (flat) search: {res['latency']['flat']['mean_ms']:.1f} ms")
    st.metric("ANN fidelity@10", f"{res['ann_fidelity@10']:.2%}", border=True,
              help="Overlap of HNSW top-10 with exact brute-force top-10.")
st.caption(f"{res['num_images']:,} photos · {res['num_products']:,} products · "
           f"{res['num_categories']} categories · {res['embedding_dim']}-d embeddings")

tab_k, tab_cat, tab_visual, tab_upload = st.tabs(
    ["Metrics by K", "By category", "Visual inspection", "Upload robustness"])

with tab_k:
    rows = [{
        "K": k,
        "Product Precision@K": overall[f"product_P@{k}"],
        "Product Recall@K (classic)": overall[f"product_R@{k}"],
        "Product Recall@K (hit rate)": overall[f"product_Hit@{k}"],
        "Product mAP@K": overall[f"product_mAP@{k}"],
        "Category Precision@K": overall[f"category_P@{k}"],
    } for k in config.EVAL_K_VALUES]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in rows[0] if c != "K"})
    st.caption("Product Precision@K is capped by how many photos a listing has (median 4-5), "
               "so it naturally falls as K grows; classic Recall@K divides by that number.")
    comparison = config.FIGURES_DIR / "model_comparison.png"
    if comparison.exists():
        st.image(str(comparison), caption="ResNet50 vs EfficientNet-B0")

with tab_cat:
    per_cat = pd.DataFrame(res["retrieval"]["per_category"]).T
    per_cat.index = per_cat.index.map(category_label)
    per_cat.index.name = "category"
    cols = {"product_Hit@1": "Product Recall@1", "product_Hit@10": "Product Recall@10",
            "product_AP@10": "Product mAP@10",
            "category_P@10": "Category P@10"}
    table = per_cat[list(cols)].rename(columns=cols).sort_values("Product Recall@1", ascending=False)
    st.dataframe(table, column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1,
                                                                         format="%.2f")
                                       for c in table.columns})
    fig = config.FIGURES_DIR / f"per_category_{model}.png"
    if fig.exists():
        st.image(str(fig))

with tab_visual:
    fig = config.FIGURES_DIR / f"visual_inspection_{model}.png"
    if fig.exists():
        st.image(str(fig), caption="Query photo (first column) and its top-5 photos. Green = same "
                 "product, blue = same category, red = different category.")

with tab_upload:
    up = res.get("upload_robustness")
    if up:
        st.caption(f"{up['n_queries']} catalog photos were randomly cropped, rotated and "
                   "colour-shifted to imitate a shopper's own photo. Does the original "
                   "product come back?")
        with st.container(horizontal=True):
            st.metric("Correct product @1", f"{up['exact_product_hit@1']:.1%}", border=True)
            st.metric("Correct product @5", f"{up['exact_product_hit@5']:.1%}", border=True)
            st.metric("Category P@10", f"{up['category_P@10']:.3f}", border=True)
            st.metric("CNN time / image", f"{up['cnn_embed_ms_per_image']:.0f} ms", border=True)
            st.metric("Search + ranking", f"{up['search_and_rank_ms']:.1f} ms", border=True)
