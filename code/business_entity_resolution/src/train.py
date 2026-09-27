"""Stage 4 — train the LightGBM pair classifier and tune the decision rule.

    python -m src.train [--rebuild]

Sample: S1 entities are bucketed by hash (0-99). Buckets < SAMPLE_PCT form the sample,
buckets < VALID_PCT of those are validation. Every candidate row of a query touching the
sample is kept, because the per-query argmax needs the query's full candidate list; rows
whose S1 is not a validation entity are used for training.
Saves <work>/model/lgbm.txt and <work>/model/decision.json.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .block import cap_candidates
from .decision import best_per_query, select_matches
from .features import FEATURES, add_context, build_features, load_attrs
from .metrics import load_truth_pairs, macro_f05

SAMPLE_PCT, VALID_PCT = 20, 5
THRESHOLDS = np.round(np.arange(0.20, 0.96, 0.025), 3)
MARGINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
PARAMS = dict(
    objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    metric=["binary_logloss", "auc"], verbose=-1, seed=0,
)


def bucket(col):
    """Stable 0-99 bucket of an id column."""
    return pl.col(col).hash(seed=42) % 100


def build_train_table(work_dir, data_dir, k):
    """Features + label + valid flag for the sampled training queries."""
    truth, _ = load_truth_pairs(data_dir)
    s1_attr, q_attr = load_attrs(work_dir, "train")
    parts = []
    for path in sorted((Path(work_dir) / "cand" / "train").glob("*.parquet")):
        t = time.time()
        c = add_context(cap_candidates(pl.read_parquet(path), k))
        touched = c.filter(bucket("s1_id") < SAMPLE_PCT)["q_id"].unique()
        c = c.filter(pl.col("q_id").is_in(touched.implode()))
        parts.append(build_features(c, s1_attr, q_attr))
        print(f"  features {path.stem}: {c.height:,} rows in {time.time() - t:.0f}s")
    feats = pl.concat(parts).join(
        truth.with_columns(label=pl.lit(1, pl.UInt8)), on=["s1_id", "q_id"], how="left"
    )
    return feats.with_columns(pl.col("label").fill_null(0), valid=bucket("s1_id") < VALID_PCT)


def tune_decision(scored, truth, valid_ids):
    """Grid-search threshold and margin for macro F0.5 on the validation S1 entities."""
    best = best_per_query(scored).filter(pl.col("s1_id").is_in(valid_ids.implode()))
    truth_v = truth.filter(pl.col("s1_id").is_in(valid_ids.implode()))
    results = [
        {"t": float(t), "margin": m, "f05": macro_f05(select_matches(best, t, m), truth_v, valid_ids)}
        for m, t in itertools.product(MARGINS, THRESHOLDS)
    ]
    res = pl.DataFrame(results).sort("f05", descending=True)
    print(res.head(8))
    return res.row(0, named=True)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM matcher")
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--k", type=int, default=config.CANDIDATE_K)
    parser.add_argument("--rebuild", action="store_true", help="recompute the cached feature table")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    model_dir = Path(args.work_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    feat_path = Path(args.work_dir) / "feat_train.parquet"
    if args.rebuild or not feat_path.exists():
        build_train_table(args.work_dir, args.data_dir, args.k).write_parquet(feat_path)
    feats = pl.read_parquet(feat_path)
    tr, va = feats.filter(~pl.col("valid")), feats.filter(pl.col("valid"))
    print(f"train rows {tr.height:,} (pos {tr['label'].mean():.3f}), valid rows {va.height:,}")

    t = time.time()
    dtrain = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["label"].to_numpy(), feature_name=FEATURES)
    dvalid = lgb.Dataset(va.select(FEATURES).to_numpy(), va["label"].to_numpy(), reference=dtrain)
    model = lgb.train(PARAMS, dtrain, num_boost_round=1500, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model.save_model(str(model_dir / "lgbm.txt"))
    print(f"trained {model.best_iteration} rounds in {time.time() - t:.0f}s")

    imp = pl.DataFrame({"feature": FEATURES, "gain": model.feature_importance("gain")})
    print(imp.sort("gain", descending=True).head(15))

    # Decision rule on validation: score every row of queries that touch a validation S1.
    truth, s1_ids = load_truth_pairs(args.data_dir)
    valid_ids = pl.DataFrame({"s1_id": s1_ids}).filter(bucket("s1_id") < VALID_PCT)["s1_id"]
    touched = va["q_id"].unique()
    rows = feats.filter(pl.col("q_id").is_in(touched.implode()))
    scored = rows.select("q_id", "s1_id").with_columns(p=model.predict(rows.select(FEATURES).to_numpy()))
    scored.write_parquet(model_dir / "scored_valid.parquet")  # reused by src.tune
    decision = tune_decision(scored, truth, valid_ids)
    decision["best_iteration"] = model.best_iteration
    print("validation macro F0.5:", round(decision["f05"], 4), decision)
    (model_dir / "decision.json").write_text(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
