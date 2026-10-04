"""
Download and extract the Stanford Online Products dataset (Song et al., CVPR 2016).

    120,053 eBay product images | 22,634 products | 12 categories | ~2.9 GB zip

Usage:
    python scripts/download_dataset.py            # download + extract into data/raw/
    python scripts/download_dataset.py --keep-zip # keep the archive afterwards

The official archive is served over FTP by Stanford (the same source that
TensorFlow Datasets uses). If the download is interrupted, re-running the
script resumes it.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

EXPECTED_BYTES = 3_083_860_082


def download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest``, resuming from a partial file when possible."""
    have = dest.stat().st_size if dest.exists() else 0
    if have == EXPECTED_BYTES:
        print(f"Archive already downloaded: {dest}")
        return

    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as resp:
        if have:
            # FTP has no Range header; skip what we already have.
            print(f"Resuming from {have / 1e9:.2f} GB")
            remaining = have
            while remaining:
                remaining -= len(resp.read(min(chunk, remaining)))
        mode = "ab" if have else "wb"
        done = have
        with open(dest, mode) as out:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if done % (100 << 20) < chunk:
                    print(f"  {done / 1e9:.2f} / {EXPECTED_BYTES / 1e9:.2f} GB", flush=True)

    if dest.stat().st_size != EXPECTED_BYTES:
        raise SystemExit(f"Download incomplete ({dest.stat().st_size:,} bytes) - re-run to resume.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--keep-zip", action="store_true", help="keep the .zip after extracting")
    args = parser.parse_args()

    raw_dir = config.SOP_ROOT.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "Stanford_Online_Products.zip"

    if (config.SOP_ROOT / "Ebay_test.txt").exists():
        print(f"Dataset already extracted at {config.SOP_ROOT}")
        return

    print(f"Downloading {config.SOP_URL}\n -> {archive}")
    download(config.SOP_URL, archive)

    print("Verifying archive ...")
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise SystemExit(f"Corrupt member in archive: {bad}. Delete the zip and re-run.")
        print(f"Extracting {len(zf.namelist()):,} files ...")
        zf.extractall(raw_dir)

    if not args.keep_zip:
        archive.unlink()
    free_gb = shutil.disk_usage(raw_dir).free / 1e9
    print(f"Done -> {config.SOP_ROOT}  ({free_gb:.1f} GB free)")


if __name__ == "__main__":
    main()
