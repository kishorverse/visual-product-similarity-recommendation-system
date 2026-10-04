"""
End-to-end pipeline: clean -> catalog DB -> embeddings -> FAISS -> evaluation.

Usage:
    python -m src.pipeline                      # everything, both backbones
    python -m src.pipeline --models resnet50    # one backbone
    python -m src.pipeline --skip-eval

Expects the extracted Stanford Online Products archive under data/raw/
(``python scripts/download_dataset.py``).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import data_cleaning, database  # noqa: E402

# torch / faiss / matplotlib are imported inside run(), not here. The cleaning
# step uses a process pool, and on Windows every worker process re-imports this
# module - importing PyTorch in 8+ workers at once exhausts memory.

logger = logging.getLogger("pipeline")


def run(models: list[str], skip_eval: bool = False) -> None:
    """Execute every stage in order, logging the time each takes."""
    config.ensure_dirs()
    timings = {}

    t0 = time.perf_counter()
    products, images, _ = data_cleaning.clean_dataset()
    database.load_catalog(products, images)
    timings["clean + load DB"] = time.perf_counter() - t0

    from src import evaluation, feature_extractor, indexer

    paths = [config.CLEAN_IMAGE_DIR / f for f in images["file_name"]]
    for model in models:
        t0 = time.perf_counter()
        embeddings = feature_extractor.extract_catalog_embeddings(model, paths)
        database.register_embeddings(model, embeddings.shape[1], images["image_id"].tolist())
        timings[f"embed ({model})"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        indexer.build_and_save_all(model)
        timings[f"index ({model})"] = time.perf_counter() - t0

    if not skip_eval:
        t0 = time.perf_counter()
        evaluation.run_evaluation(models)
        timings["evaluation"] = time.perf_counter() - t0

    logger.info("Pipeline finished:")
    for stage, secs in timings.items():
        logger.info("  %-28s %6.1fs", stage, secs)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the full visual-similarity pipeline")
    parser.add_argument("--models", nargs="+", default=list(config.MODEL_DIMS),
                        choices=list(config.MODEL_DIMS))
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    try:
        run(args.models, args.skip_eval)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
