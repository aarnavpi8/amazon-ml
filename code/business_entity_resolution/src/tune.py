"""Tune the decision rule (threshold, margin) at test-like decoy density.

The test set has ~1.9x more decoy S2/S3 records (records matching no S1) per S1 entity than
train, so a threshold tuned on train validation is too lenient. Decoy queries in the
validation scores are replicated until their density matches the test set, then
(threshold, margin) are grid-searched for macro F0.5.

    python -m src.tune [--model-dir work/model]
Reads/writes <model-dir>/scored_valid.parquet and <model-dir>/decision.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .decision import best_per_query, select_matches
from .features import FEATURES
from .metrics import load_truth_pairs, macro_f05

VALID_PCT = 5  # must match train.py
THRESHOLDS = np.round(np.arange(0.30, 0.99, 0.025), 3)
MARGINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def bucket(col):
    """Stable 0-99 bucket of an id column (same as train.py)."""
    return pl.col(col).hash(seed=42) % 100


def n_rows(path):
    return pl.scan_parquet(path).select(pl.len()).collect().item()


def estimate_decoy_boost(work_dir, truth):
    """Test decoys per S1 divided by train decoys per S1 (true matches per S1 assumed equal)."""
    count = lambda split: (
        n_rows(config.norm_path(work_dir, split, "source1")),
        sum(n_rows(config.norm_path(work_dir, split, s)) for s in ("source2", "source3")),
    )
    tr_s1, tr_q = count("train")
    te_s1, te_q = count("test")
    matches_per_s1 = truth.height / tr_s1
    train_decoys = tr_q / tr_s1 - matches_per_s1
    test_decoys = te_q / te_s1 - matches_per_s1
    return test_decoys / train_decoys


def boost_decoys(scored, truth, boost):
    """Replicate decoy queries (no true S1 at all) so their density is multiplied by `boost`."""
    decoys = scored.join(truth.select("q_id").unique(), on="q_id", how="anti")
    extra, copies = [], boost - 1.0
    i = 0
    while copies > 1e-9:
        frac = min(copies, 1.0)
        keep = decoys.filter((pl.col("q_id").hash(seed=100 + i) % 10_000) < frac * 10_000)
        extra.append(keep.with_columns(pl.col("q_id") + f"#dup{i}"))
        copies -= frac
        i += 1
    return pl.concat([scored, *extra])


def score_validation(work_dir, model):
    """Model scores for every candidate row of queries touching a validation S1."""
    feats = pl.read_parquet(Path(work_dir) / "feat_train.parquet")
    touched = feats.filter(pl.col("valid"))["q_id"].unique()
    rows = feats.filter(pl.col("q_id").is_in(touched.implode()))
    return rows.select("q_id", "s1_id").with_columns(p=model.predict(rows.select(FEATURES).to_numpy()))


def tune(scored, truth, valid_ids):
    """Grid-search (threshold, margin); returns the best row and the full table."""
    best = best_per_query(scored).filter(pl.col("s1_id").is_in(valid_ids.implode()))
    truth_v = truth.filter(pl.col("s1_id").is_in(valid_ids.implode()))
    res = pl.DataFrame([
        {"t": float(t), "margin": m, "f05": macro_f05(select_matches(best, t, m), truth_v, valid_ids)}
        for m, t in itertools.product(MARGINS, THRESHOLDS)
    ]).sort("f05", descending=True)
    return res.row(0, named=True), res


def main():
    parser = argparse.ArgumentParser(description="Tune decision rule at test decoy density")
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--model-dir", default=None, help="default: <work>/model")
    parser.add_argument("--boost", type=float, default=None, help="override the estimated decoy boost")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    model_dir = Path(args.model_dir or Path(args.work_dir) / "model")

    truth, s1_ids = load_truth_pairs(args.data_dir)
    scored_path = model_dir / "scored_valid.parquet"
    if scored_path.exists():
        scored = pl.read_parquet(scored_path, columns=["q_id", "s1_id", "p"])
    else:
        scored = score_validation(args.work_dir, lgb.Booster(model_file=str(model_dir / "lgbm.txt")))
        scored.write_parquet(scored_path)
    valid_ids = pl.DataFrame({"s1_id": s1_ids}).filter(bucket("s1_id") < VALID_PCT)["s1_id"]

    boost = args.boost or estimate_decoy_boost(args.work_dir, truth)
    plain, _ = tune(scored, truth, valid_ids)
    boosted, table = tune(boost_decoys(scored, truth, boost), truth, valid_ids)
    print(f"decoy boost {boost:.2f}")
    print(table.head(8))
    print(f"train density: t={plain['t']} m={plain['margin']} F0.5={plain['f05']:.4f}")
    print(f"test density : t={boosted['t']} m={boosted['margin']} F0.5={boosted['f05']:.4f}")

    decision = json.loads((model_dir / "decision.json").read_text()) if (model_dir / "decision.json").exists() else {}
    decision.update(t=boosted["t"], margin=boosted["margin"], f05_test_density=boosted["f05"],
                    f05_train_density=plain["f05"], decoy_boost=boost)
    (model_dir / "decision.json").write_text(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
