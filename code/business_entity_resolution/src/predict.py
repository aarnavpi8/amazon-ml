"""Stage 5 — score the test candidates and write the submission files.

    python -m src.predict
Countries are processed one at a time (queries never cross countries), so memory stays low.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

from . import config
from .block import cap_candidates
from .decision import best_per_query, select_matches
from .features import FEATURES, add_context, build_features, load_attrs
from .submission import test_s1_ids, write_outputs


def main():
    parser = argparse.ArgumentParser(description="Predict test matches")
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--out-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=config.CANDIDATE_K)
    parser.add_argument("--model-dir", default=None, help="default: <work>/model")
    parser.add_argument("--reuse-scores", action="store_true",
                        help="skip feature building/scoring; re-apply the decision to saved test scores")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    model_dir = Path(args.model_dir or Path(args.work_dir) / "model")
    decision = json.loads((model_dir / "decision.json").read_text())
    print(f"decision: t={decision['t']} margin={decision['margin']}")
    scored_dir = model_dir / "scored_test"
    scored_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_scores:
        model = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
        s1_attr, q_attr = load_attrs(args.work_dir, "test")

    matches, cands = [], []
    for path in sorted((Path(args.work_dir) / "cand" / "test").glob("*.parquet")):
        t = time.time()
        if args.reuse_scores:
            scored = pl.read_parquet(scored_dir / path.name)
            c = scored
        else:
            c = add_context(cap_candidates(pl.read_parquet(path), args.k))
            f = build_features(c, s1_attr, q_attr)
            scored = f.select("q_id", "s1_id").with_columns(p=model.predict(f.select(FEATURES).to_numpy()))
            scored.write_parquet(scored_dir / path.name)  # reused by --reuse-scores
        m = select_matches(best_per_query(scored), decision["t"], decision["margin"])
        print(f"  {path.stem}: {c.height:,} candidate pairs -> {m.height:,} matches "
              f"({m['s1_id'].n_unique():,} S1 matched) in {time.time() - t:.0f}s")
        matches.append(m)
        cands.append(c.select("s1_id", "q_id"))
    write_outputs(pl.concat(matches), pl.concat(cands), test_s1_ids(args.data_dir), args.out_dir)


if __name__ == "__main__":
    main()
