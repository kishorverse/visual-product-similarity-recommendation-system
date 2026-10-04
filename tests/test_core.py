"""
Unit tests for the pieces that do not need the CNN or the real dataset:
metrics, SOP index parsing, duplicate detection, FAISS indexing and the
SQLite catalog.

Run:  pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import database  # noqa: E402
from src.data_cleaning import (colour_thumbnail, difference_hash, find_duplicates,  # noqa: E402
                               hamming_distance, load_sop_index, sample_products)
from src.evaluation import (average_precision_at_k, batch_metrics,  # noqa: E402
                            leave_one_out_search, precision_at_k, recall_at_k,
                            retrieval_metrics)
from src.indexer import build_index  # noqa: E402


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_precision_and_recall():
    rel = [True, False, True, True, False]
    assert precision_at_k(rel, 1) == 1.0
    assert precision_at_k(rel, 5) == pytest.approx(0.6)
    assert recall_at_k(rel, 5, total_relevant=6) == pytest.approx(0.5)
    assert recall_at_k(rel, 5, total_relevant=0) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(rel, 0)


def test_average_precision():
    # Relevant at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert average_precision_at_k([True, False, True], 3, 2) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_at_k([False, False], 2, 3) == 0.0


def test_batch_metrics_match_scalar_versions():
    rng = np.random.default_rng(0)
    rel = rng.random((50, 10)) > 0.6
    total = rng.integers(0, 8, 50)
    out = batch_metrics(rel, total, 5)
    for i in range(50):
        assert out["P"][i] == pytest.approx(precision_at_k(rel[i], 5))
        assert out["R"][i] == pytest.approx(recall_at_k(rel[i], 5, int(total[i])))
        assert out["AP"][i] == pytest.approx(average_precision_at_k(rel[i], 5, int(total[i])))
        assert out["Hit"][i] == float(rel[i, :5].any())


def test_retrieval_metrics_product_and_category_levels():
    # Two products per category, two photos each.
    products = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    categories = np.array(["x", "x", "x", "x", "y", "y", "y", "y"])
    # Each photo's top-2: its sibling photo first, then a photo of the other product in its category.
    neighbours = np.array([[1, 2], [0, 3], [3, 0], [2, 1], [5, 6], [4, 7], [7, 4], [6, 5]])
    summary, per_query = retrieval_metrics(neighbours, products, categories, [1, 2])
    o = summary["overall"]
    assert o["product_Hit@1"] == 1.0
    assert o["product_P@2"] == 0.5
    assert o["product_R@1"] == 1.0
    assert o["category_P@2"] == 1.0
    assert len(per_query) == 8


# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
def test_load_sop_index_and_sampling(tmp_path):
    (tmp_path / "Ebay_test.txt").write_text(
        "image_id class_id super_class_id path\n"
        "1 10 1 bicycle_final/111_0.JPG\n"
        "2 10 1 bicycle_final/111_1.JPG\n"
        "3 11 4 coffee_maker_final/222_0.JPG\n"
        "4 12 4 coffee_maker_final/333_0.JPG\n"
    )
    idx = load_sop_index(tmp_path, "test")
    assert idx["category"].tolist() == ["bicycle", "bicycle", "coffee_maker", "coffee_maker"]
    assert idx["product_id"].tolist() == ["111", "111", "222", "333"]
    assert idx["view"].tolist() == [0, 1, 0, 0]

    sampled = sample_products(idx, max_per_category=1)
    assert sampled.groupby("category")["product_id"].nunique().max() == 1
    assert len(sampled[sampled["product_id"] == "111"]) in (0, 2)   # all photos of a product kept together
    with pytest.raises(ValueError):
        load_sop_index(tmp_path, "validation")


def _noise_image(seed, size=(160, 120)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_dhash_detects_resized_copy_but_not_different_image():
    img, other = _noise_image(0), _noise_image(1)
    h1, h2, h3 = difference_hash(img), difference_hash(img.resize((320, 240))), difference_hash(other)
    assert hamming_distance(h1, h2) <= 4
    assert hamming_distance(h1, h3) > 10


def test_find_duplicates_keeps_colour_variants():
    base = Image.new("RGB", (64, 64), "white")
    base.paste(Image.new("RGB", (30, 30), (200, 30, 30)), (17, 17))   # red square
    variant = Image.new("RGB", (64, 64), "white")
    variant.paste(Image.new("RGB", (30, 30), (30, 30, 200)), (17, 17))  # blue square, same shape
    images = [base, base.resize((128, 128)).resize((64, 64)), variant, _noise_image(5, (64, 64))]
    hashes = np.array([difference_hash(i) for i in images], dtype=np.uint64)
    thumbs = np.stack([colour_thumbnail(i) for i in images])
    dups = find_duplicates(hashes, thumbs)
    assert dups == {1: 0}          # the re-encoded copy is a duplicate; the blue variant is not


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def test_gem_pooling_interpolates_between_avg_and_max():
    import torch
    from src.feature_extractor import GeM

    x = torch.rand(2, 8, 7, 7) + 0.01
    avg = x.mean(dim=(2, 3))
    mx = x.amax(dim=(2, 3))
    assert torch.allclose(GeM(p=1.0)(x).flatten(1), avg, atol=1e-5)
    gem3 = GeM(p=3.0)(x).flatten(1)
    assert torch.all(gem3 >= avg - 1e-6) and torch.all(gem3 <= mx + 1e-6)


# --------------------------------------------------------------------------- #
# FAISS
# --------------------------------------------------------------------------- #
def _clustered_embeddings(n_per=20, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, dim))
    x = np.vstack([c + 0.1 * rng.normal(size=(n_per, dim)) for c in centres]).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels = np.repeat(["a", "b", "c"], n_per)
    return x, labels


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_index_returns_cosine_neighbours(index_type):
    x, labels = _clustered_embeddings()
    index = build_index(x, index_type)
    scores, rows = index.search(x[:1], 1)
    assert rows[0, 0] == 0 and scores[0, 0] == pytest.approx(1.0, abs=1e-4)

    neighbours = leave_one_out_search(index, x, 5)
    assert 0 not in neighbours[0]
    summary, _ = retrieval_metrics(neighbours, labels, labels, [5])
    assert summary["overall"]["category_P@5"] == 1.0


def test_unknown_index_type():
    with pytest.raises(ValueError):
        build_index(np.zeros((2, 4), dtype="float32"), "ivf-magic")


# --------------------------------------------------------------------------- #
# SQLite catalog
# --------------------------------------------------------------------------- #
@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test.db"
    products = pd.DataFrame({
        "product_id": ["111", "222", "333"], "class_id": [10, 11, 12],
        "category": ["chair", "chair", "mug"],
        "product_name": ["Chair #0001", "Chair #0002", "Mug #0001"],
        "price": [999, 4999, 299], "rating": [4.0, 3.5, 4.8], "in_stock": [True, False, True],
    })
    images = pd.DataFrame({
        "image_id": [1, 2, 3, 4], "product_id": ["111", "111", "222", "333"], "view": [0, 1, 0, 0],
        "file_name": ["chair/111_0.jpg", "chair/111_1.jpg", "chair/222_0.jpg", "mug/333_0.jpg"],
        "width": [320] * 4, "height": [320] * 4, "orig_width": [500] * 4, "orig_height": [500] * 4,
        "dhash": ["0"] * 4,
    })
    database.load_catalog(products, images, path)
    database.register_embeddings("resnet50", 2048, images["image_id"].tolist(), path)
    return path


def test_catalog_filters_and_pagination(db):
    page, total = database.get_catalog_page(categories=["chair"], db_path=db)
    assert total == 2 and set(page["category"]) == {"chair"}
    assert page.set_index("product_id").loc["111", "num_images"] == 2
    assert page.set_index("product_id").loc["111", "image_file"] == "chair/111_0.jpg"
    _, total = database.get_catalog_page(in_stock_only=True, db_path=db)
    assert total == 2
    _, total = database.get_catalog_page(price_range=(500, 3000), db_path=db)
    assert total == 1
    page, total = database.get_catalog_page(limit=2, offset=2, db_path=db)
    assert total == 3 and len(page) == 1


def test_row_mapping_and_product_images(db):
    mapping = database.get_row_mapping("resnet50", db)
    assert mapping["vector_row"].tolist() == [0, 1, 2, 3]
    assert mapping["product_id"].tolist() == ["111", "111", "222", "333"]
    assert database.get_product_images("111", db) == ["chair/111_0.jpg", "chair/111_1.jpg"]


def test_foreign_keys_enforced(db):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        with database.get_connection(db) as conn:
            conn.execute("INSERT INTO product_images (image_id, product_id, view_index, image_file) "
                         "VALUES (99, 'missing', 0, 'x.jpg')")


def test_search_log_roundtrip(db):
    database.log_search("upload", "resnet50", 10, 10, 12.5, db_path=db)
    stats = database.get_search_stats(db)
    assert stats.loc[0, "searches"] == 1
