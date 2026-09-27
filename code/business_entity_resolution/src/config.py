"""Paths shared by every pipeline stage.

Directories can be overridden with environment variables (handy on Colab):
    ER_DATA_DIR    folder containing train/ and test/ with the challenge TSVs
    ER_WORK_DIR    folder for intermediate artefacts (parquet files, dictionaries, models)
    ER_OUTPUT_DIR  folder for matching_results.tsv / candidate_pairs.tsv
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR.parents[1] / "dataset" / "student_resource" / "dataset"

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", DEFAULT_DATA_DIR))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", PROJECT_DIR / "work"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", PROJECT_DIR.parents[1] / "output"))

CANDIDATE_K = 5  # S1 candidates kept per S2/S3 record (97.9% pair recall on train)

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")


def source_path(data_dir, split, source):
    """Path of one challenge TSV, e.g. <data>/train/train_source2.tsv."""
    return Path(data_dir) / split / f"{split}_{source}.tsv"


def norm_path(work_dir, split, source):
    """Path of the normalised parquet written for one source file."""
    return Path(work_dir) / "norm" / f"{split}_{source}.parquet"
