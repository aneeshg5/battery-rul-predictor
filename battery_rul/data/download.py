"""Download and extract the NASA Randomized Battery Usage 1: Random Walk dataset."""

import hashlib
import json
import logging
import zipfile
from pathlib import Path

import requests

from battery_rul.config import BATTERIES, RAW_DIR

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://data.nasa.gov/docs/legacy/ames/"
    "1.Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post.zip"
)
MAT_PREFIX = "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(raw_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    """Download RW9-RW12 .mat files into raw_dir, skipping re-download if already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = [raw_dir / f"{name}.mat" for name in BATTERIES]
    checksum_path = raw_dir / "checksums.json"

    if not force and all(t.exists() for t in targets) and checksum_path.exists():
        logger.info("Raw battery files already present in %s, skipping download", raw_dir)
        return targets

    zip_path = raw_dir / "_dataset.zip"
    logger.info("Downloading NASA battery dataset from %s", DATASET_URL)
    response = requests.get(DATASET_URL, stream=True, timeout=300)
    response.raise_for_status()
    with zip_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    checksums: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in BATTERIES:
            member = f"{MAT_PREFIX}{name}.mat"
            target = raw_dir / f"{name}.mat"
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            checksums[name] = _sha256(target)

    checksum_path.write_text(json.dumps(checksums, indent=2))
    zip_path.unlink()
    logger.info("Saved raw battery files to %s", raw_dir)
    return targets
