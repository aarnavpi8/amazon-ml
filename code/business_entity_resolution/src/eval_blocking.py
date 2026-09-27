"""Blocking quality on the training split (needs `python -m src.block --split train` first).

    python -m src.eval_blocking [--ks 1 2 3 5 8 10 15]

Reports, for each K (candidates kept per S2/S3 query):
  pair recall, recall by segment, candidates per S1, and the oracle macro F0.5
  (the score a perfect matcher would reach on this candidate set = the blocking ceiling).
"""
import argparse
import sys

import polars as pl
from rapidfuzz import fuzz, process, utils

from . import config
from .block import load_candidates
from .metrics import load_truth_pairs, macro_f05


def truth_segments(work_dir, truth):
    """Label every true pair with the segments we care about."""
    s1 = pl.read_parquet(config.norm_path(work_dir, "train", "source1"),
                         columns=["entity_id", "country", "name_core", "addr_norm"])
    q = pl.concat([
        pl.read_parquet(config.norm_path(work_dir, "train", s), columns=["entity_id", "name_core", "name_indic", "addr_state", "addr_norm"])
        for s in ("source2", "source3")
    ])
    t = (
        truth.join(s1.rename({"entity_id": "s1_id", "name_core": "n1", "addr_norm": "a1"}), on="s1_id")
        .join(q.rename({"entity_id": "q_id", "name_core": "n2"}), on="q_id")
    )
    sim = process.cpdist(t["n1"].to_list(), t["n2"].to_list(), scorer=fuzz.token_set_ratio,
                         processor=utils.default_process, workers=-1)
    return t.with_columns(name_sim=sim).select(
        "s1_id", "q_id", "country", "n1", "n2", "a1", a2="addr_norm",
        cross_script=pl.col("name_indic"),
        invented_name=pl.col("name_sim") < 40,
        no_state=pl.col("addr_state").is_null(),
        empty_addr=pl.col("addr_norm") == "",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate blocking on train")
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # polars tables use box-drawing characters

    truth, s1_ids = load_truth_pairs(args.data_dir)
    cands = load_candidates(args.work_dir, "train").with_columns(
        rank=pl.col("block_score").rank("ordinal", descending=True).over("q_id")
    )
    seg = truth_segments(args.work_dir, truth).join(
        cands.select("s1_id", "q_id", "rank", "by_name", "by_addr", "by_exact"), on=["s1_id", "q_id"], how="left"
    )
    n_s1 = len(s1_ids)
    print(f"true pairs: {len(truth):,}   stored candidate pairs: {cands.height:,}")

    rows = []
    for k in args.ks:
        hit = seg.filter(pl.col("rank") <= k).select("s1_id", "q_id")
        rows.append({
            "K": k,
            "pair_recall": round(hit.height / seg.height, 4),
            "cands_per_S1": round(cands.filter(pl.col("rank") <= k).height / n_s1, 1),
            "oracle_F05": round(macro_f05(hit, truth, s1_ids), 4),
        })
    print(pl.DataFrame(rows))

    k = max(args.ks)
    found = (pl.col("rank") <= k).fill_null(False)
    print(f"\nrecall by segment at K={k}:")
    for col in ("country", "cross_script", "invented_name", "no_state", "empty_addr"):
        print(seg.group_by(col).agg(pairs=pl.len(), recall=found.mean().round(4)).sort(col))

    print(f"\nwhich pass found the true pairs (K={k}):")
    print(seg.filter(found).select(
        name=pl.col("by_name").mean(), addr=pl.col("by_addr").mean(), exact=pl.col("by_exact").mean(),
        only_addr=(pl.col("by_addr") & ~pl.col("by_name")).mean(),
        only_exact=(pl.col("by_exact") & ~pl.col("by_name") & ~pl.col("by_addr")).mean(),
    ))

    with pl.Config(fmt_str_lengths=60, tbl_width_chars=250, tbl_rows=25):
        print("\nsample of missed true pairs:")
        print(seg.filter(~found).sample(25, seed=0).select("country", "n1", "n2", "a1", "a2"))


if __name__ == "__main__":
    main()
