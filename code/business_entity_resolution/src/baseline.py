"""Baseline matcher (submission #1): a tuned rule on the blocking similarities.

    score = w * name_cos + (1 - w) * addr_cos
Each S2/S3 record is assigned to its best-scoring candidate S1 if score >= t and it beats
the runner-up by at least `margin` (every S2/S3 record belongs to at most one S1).
(w, t, margin) are grid-searched for macro F0.5 on the training split.

    python -m src.baseline            # tune on train, then write output/ for test
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from . import config
from .block import cap_candidates, load_candidates
from .metrics import load_truth_pairs, macro_f05
from .submission import test_s1_ids, write_outputs

WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]
MARGINS = [0.0, 0.05, 0.1, 0.2]
THRESHOLDS = np.round(np.arange(0.40, 0.96, 0.025), 3)
TUNE_S1_SAMPLE = 300_000


def best_per_query(cands, w):
    """For every query: its best S1 under weight w, the score and the margin over the runner-up."""
    scored = cands.with_columns(score=w * pl.col("name_cos") + (1 - w) * pl.col("addr_cos"))
    return scored.group_by("q_id").agg(
        s1_id=pl.col("s1_id").sort_by("score", descending=True).first(),
        score=pl.col("score").max(),
        second=pl.when(pl.len() > 1).then(pl.col("score").top_k(2).min()).otherwise(0.0),
    ).with_columns(margin=pl.col("score") - pl.col("second"))


def predict(best, t, margin):
    """Accepted (s1_id, q_id) matches."""
    return best.filter(pl.col("score") >= t, pl.col("margin") >= margin).select("s1_id", "q_id")


def tune(work_dir, data_dir, k):
    """Grid-search (w, t, margin) on a sample of train S1 entities."""
    truth, s1_ids = load_truth_pairs(data_dir)
    cands = cap_candidates(load_candidates(work_dir, "train"), k).select("q_id", "s1_id", "name_cos", "addr_cos")
    sample = s1_ids.sample(TUNE_S1_SAMPLE, seed=0)
    truth_s = truth.filter(pl.col("s1_id").is_in(sample.implode()))

    results = []
    for w in WEIGHTS:
        best = best_per_query(cands, w).filter(pl.col("s1_id").is_in(sample.implode()))
        for m, t in itertools.product(MARGINS, THRESHOLDS):
            results.append({"w": w, "margin": m, "t": float(t),
                            "f05": macro_f05(predict(best, t, m), truth_s, sample)})
        top = max(r["f05"] for r in results if r["w"] == w)
        print(f"  w={w}: best F0.5 {top:.4f}")
    res = pl.DataFrame(results).sort("f05", descending=True)
    print(res.head(10))
    return res.row(0, named=True)


def main():
    parser = argparse.ArgumentParser(description="Tuned-rule baseline")
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--out-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=config.CANDIDATE_K)
    parser.add_argument("--step", choices=["tune", "predict", "all"], default="all")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    params_path = Path(args.work_dir, "baseline_params.json")
    if args.step in ("tune", "all"):
        params = tune(args.work_dir, args.data_dir, args.k)
        print("chosen:", params)
        params_path.write_text(json.dumps(params, indent=1))
    if args.step == "tune":
        return
    params = json.loads(params_path.read_text())

    cands = cap_candidates(load_candidates(args.work_dir, "test"), args.k)
    best = best_per_query(cands, params["w"])
    matches = predict(best, params["t"], params["margin"])
    write_outputs(matches, cands.select("s1_id", "q_id"), test_s1_ids(args.data_dir), args.out_dir)


if __name__ == "__main__":
    main()
