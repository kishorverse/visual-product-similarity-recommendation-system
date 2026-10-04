"""
Visual Product Similarity - Streamlit front end.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.common import render_sidebar  # noqa: E402

st.set_page_config(
    page_title="Visual product search",
    page_icon=":material/image_search:",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent

page = st.navigation(
    [
        st.Page(APP_DIR / "app_pages" / "image_search.py", title="Image search",
                icon=":material/image_search:", default=True),
        st.Page(APP_DIR / "app_pages" / "similar_items.py", title="Similar items",
                icon=":material/grid_view:"),
        st.Page(APP_DIR / "app_pages" / "evaluation.py", title="Evaluation",
                icon=":material/query_stats:"),
        st.Page(APP_DIR / "app_pages" / "about.py", title="How it works",
                icon=":material/info:"),
    ],
    position="top",
)

render_sidebar()
page.run()
