"""
SQLite product catalog.

Schema (3NF):

    categories (category_id PK, category_name UNIQUE)
         | 1:N
    products (product_id PK = eBay item id, category_id FK, product_name, price, rating, in_stock)
         | 1:N
    product_images (image_id PK, product_id FK, view_index, image_file UNIQUE, size, hash)
         | 1:N
    image_embeddings (model_id FK, vector_row, image_id FK)   -> FAISS row -> image -> product
         | N:1
    embedding_models (model_id PK, model_name UNIQUE, embedding_dim, num_vectors)

    search_logs (search_id PK, query_type, model_name, top_k, latency_ms, ...)

A product (one eBay listing) has several photos. FAISS stores one vector per
photo; ``image_embeddings`` maps a FAISS row back to the photo and its product.

Usage:
    python -m src.database          # (re)build the catalog from processed CSVs
"""

import logging
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS products (
    product_id    TEXT PRIMARY KEY,               -- eBay item id
    sop_class_id  INTEGER NOT NULL UNIQUE,        -- SOP benchmark class id
    product_name  TEXT NOT NULL,
    category_id   INTEGER NOT NULL REFERENCES categories(category_id),
    price         INTEGER NOT NULL CHECK (price >= 0),
    rating        REAL CHECK (rating BETWEEN 0 AND 5),
    in_stock      INTEGER NOT NULL DEFAULT 1 CHECK (in_stock IN (0, 1))
);

CREATE TABLE IF NOT EXISTS product_images (
    image_id      INTEGER PRIMARY KEY,            -- SOP image id
    product_id    TEXT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    view_index    INTEGER NOT NULL,
    image_file    TEXT NOT NULL UNIQUE,           -- relative to data/processed/images
    width         INTEGER,
    height        INTEGER,
    orig_width    INTEGER,
    orig_height   INTEGER,
    image_hash    TEXT
);

CREATE TABLE IF NOT EXISTS embedding_models (
    model_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT NOT NULL UNIQUE,
    embedding_dim INTEGER NOT NULL,
    num_vectors   INTEGER NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS image_embeddings (
    model_id    INTEGER NOT NULL REFERENCES embedding_models(model_id) ON DELETE CASCADE,
    vector_row  INTEGER NOT NULL,
    image_id    INTEGER NOT NULL REFERENCES product_images(image_id) ON DELETE CASCADE,
    PRIMARY KEY (model_id, vector_row),
    UNIQUE (model_id, image_id)
);

CREATE TABLE IF NOT EXISTS search_logs (
    search_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    query_type      TEXT NOT NULL,                -- 'upload' | 'catalog'
    model_name      TEXT NOT NULL,
    top_k           INTEGER NOT NULL,
    num_results     INTEGER NOT NULL,
    latency_ms      REAL NOT NULL,
    category_filter TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for the filter / lookup paths used by the app.
CREATE INDEX IF NOT EXISTS idx_products_category   ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_price      ON products(price);
CREATE INDEX IF NOT EXISTS idx_products_stock      ON products(in_stock);
CREATE INDEX IF NOT EXISTS idx_images_product      ON product_images(product_id, view_index);
CREATE INDEX IF NOT EXISTS idx_embeddings_image    ON image_embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_search_logs_time    ON search_logs(created_at);
"""

# A product with its category name and its main (lowest view index) photo.
PRODUCT_SELECT = """
SELECT p.product_id, p.product_name, c.category_name AS category,
       p.price, p.rating, p.in_stock,
       (SELECT i.image_file FROM product_images i
         WHERE i.product_id = p.product_id ORDER BY i.view_index LIMIT 1) AS image_file,
       (SELECT COUNT(*) FROM product_images i WHERE i.product_id = p.product_id) AS num_images
FROM products p
JOIN categories c ON c.category_id = p.category_id
"""


@contextmanager
def get_connection(db_path: Path = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection with FK enforcement; commit on success, roll back on error."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        logger.exception("SQLite error - transaction rolled back")
        raise
    finally:
        conn.close()


def init_schema(db_path: Path = config.DB_PATH) -> None:
    """Create all tables and indexes if they do not exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def load_catalog(products: pd.DataFrame, images: pd.DataFrame, db_path: Path = config.DB_PATH) -> int:
    """
    Replace the catalog with ``products`` and their ``images``.

    Embedding mappings reference images, so they are cleared too and must be
    re-registered after features are extracted.
    """
    init_schema(db_path)
    with get_connection(db_path) as conn:
        for table in ("image_embeddings", "embedding_models", "product_images", "products", "categories"):
            conn.execute(f"DELETE FROM {table}")

        categories = sorted(products["category"].unique())
        conn.executemany("INSERT INTO categories (category_name) VALUES (?)", [(c,) for c in categories])
        category_ids = dict(conn.execute("SELECT category_name, category_id FROM categories"))

        conn.executemany(
            """INSERT INTO products (product_id, sop_class_id, product_name, category_id,
                                     price, rating, in_stock)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(str(r.product_id), int(r.class_id), r.product_name, category_ids[r.category],
              int(r.price), float(r.rating), int(bool(r.in_stock)))
             for r in products.itertuples(index=False)],
        )
        conn.executemany(
            """INSERT INTO product_images (image_id, product_id, view_index, image_file, width,
                                           height, orig_width, orig_height, image_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(int(r.image_id), str(r.product_id), int(r.view), r.file_name, int(r.width),
              int(r.height), int(r.orig_width), int(r.orig_height), r.dhash)
             for r in images.itertuples(index=False)],
        )
    logger.info("Loaded %d products / %d images in %d categories into %s",
                len(products), len(images), len(categories), db_path)
    return len(products)


def register_embeddings(
    model_name: str,
    embedding_dim: int,
    image_ids: Sequence[int],
    db_path: Path = config.DB_PATH,
) -> None:
    """Record which image sits at each row of the embedding matrix / FAISS index."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM embedding_models WHERE model_name = ?", (model_name,))
        cur = conn.execute(
            "INSERT INTO embedding_models (model_name, embedding_dim, num_vectors) VALUES (?, ?, ?)",
            (model_name, embedding_dim, len(image_ids)),
        )
        model_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO image_embeddings (model_id, vector_row, image_id) VALUES (?, ?, ?)",
            [(model_id, row, int(iid)) for row, iid in enumerate(image_ids)],
        )


def get_row_mapping(model_name: str, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """One row per FAISS vector (ordered by vector_row) with image + product metadata."""
    query = """
        SELECT e.vector_row, i.image_id, i.image_file, i.view_index,
               p.product_id, p.sop_class_id, p.product_name, c.category_name AS category,
               p.price, p.rating, p.in_stock
        FROM image_embeddings e
        JOIN embedding_models m ON m.model_id = e.model_id
        JOIN product_images i   ON i.image_id = e.image_id
        JOIN products p         ON p.product_id = i.product_id
        JOIN categories c       ON c.category_id = p.category_id
        WHERE m.model_name = ?
        ORDER BY e.vector_row
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=(model_name,))


def get_categories(db_path: Path = config.DB_PATH) -> list[str]:
    """All category names, alphabetically."""
    with get_connection(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT category_name FROM categories ORDER BY 1")]


def get_price_bounds(db_path: Path = config.DB_PATH) -> tuple[int, int]:
    """(min, max) price across the catalog."""
    with get_connection(db_path) as conn:
        low, high = conn.execute("SELECT MIN(price), MAX(price) FROM products").fetchone()
    return int(low or 0), int(high or 0)


def _filter_clause(
    categories: Optional[Sequence[str]],
    price_range: Optional[tuple[int, int]],
    in_stock_only: bool,
) -> tuple[str, list]:
    """Build a parameterised WHERE clause for the catalog filters."""
    clauses, params = [], []
    if categories:
        clauses.append(f"c.category_name IN ({','.join('?' * len(categories))})")
        params.extend(categories)
    if price_range:
        clauses.append("p.price BETWEEN ? AND ?")
        params.extend([int(price_range[0]), int(price_range[1])])
    if in_stock_only:
        clauses.append("p.in_stock = 1")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def get_catalog_page(
    categories: Optional[Sequence[str]] = None,
    price_range: Optional[tuple[int, int]] = None,
    in_stock_only: bool = False,
    limit: int = config.CATALOG_PAGE_SIZE,
    offset: int = 0,
    db_path: Path = config.DB_PATH,
) -> tuple[pd.DataFrame, int]:
    """One page of filtered products (with main photo) plus the total count, for pagination."""
    where, params = _filter_clause(categories, price_range, in_stock_only)
    with get_connection(db_path) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM products p JOIN categories c ON c.category_id = p.category_id{where}",
            params,
        ).fetchone()[0]
        page = pd.read_sql_query(
            f"{PRODUCT_SELECT}{where} ORDER BY c.category_name, p.product_name LIMIT ? OFFSET ?",
            conn, params=[*params, limit, offset],
        )
    return page, int(total)


def get_product(product_id: str, db_path: Path = config.DB_PATH) -> Optional[dict]:
    """A single product as a dict, or None."""
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(f"{PRODUCT_SELECT} WHERE p.product_id = ?", conn, params=(str(product_id),))
    return None if df.empty else df.iloc[0].to_dict()


def get_product_images(product_id: str, db_path: Path = config.DB_PATH) -> list[str]:
    """All photo files of a product, in view order."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT image_file FROM product_images WHERE product_id = ? ORDER BY view_index",
            (str(product_id),),
        ).fetchall()
    return [r[0] for r in rows]


def log_search(
    query_type: str,
    model_name: str,
    top_k: int,
    num_results: int,
    latency_ms: float,
    category_filter: Optional[str] = None,
    db_path: Path = config.DB_PATH,
) -> None:
    """Persist one search event (feeds the app's usage panel). Never raises."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO search_logs (query_type, model_name, top_k, num_results,
                                            latency_ms, category_filter)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (query_type, model_name, top_k, num_results, latency_ms, category_filter),
            )
    except sqlite3.Error:
        logger.warning("Could not write search log", exc_info=True)


def get_search_stats(db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """Aggregate search usage by query type and model."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            """SELECT query_type, model_name, COUNT(*) AS searches,
                      ROUND(AVG(latency_ms), 2) AS avg_latency_ms
               FROM search_logs GROUP BY query_type, model_name ORDER BY searches DESC""",
            conn,
        )


def main() -> None:
    """CLI entry point: build the catalog tables from the processed CSVs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not config.PRODUCTS_CSV.exists():
        raise SystemExit("Processed CSVs not found - run `python -m src.data_cleaning` first.")
    load_catalog(pd.read_csv(config.PRODUCTS_CSV, dtype={"product_id": str}),
                 pd.read_csv(config.IMAGES_CSV, dtype={"product_id": str}))


if __name__ == "__main__":
    main()
