"""Write the two challenge output files in the exact required format."""
from pathlib import Path

import polars as pl

from . import config


def _id_lists(pairs, s1_ids, col):
    """One row per S1 id with a comma-joined, de-duplicated list of matched ids (may be empty)."""
    lists = pairs.unique(["s1_id", "q_id"]).group_by("s1_id").agg(
        pl.col("q_id").sort().str.join(",").alias(col)
    )
    return (
        pl.DataFrame({"source1_entity_id": s1_ids})
        .join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
        .with_columns(pl.col(col).fill_null(""))
    )


def write_outputs(matches, candidates, s1_ids, out_dir=config.OUTPUT_DIR):
    """Write matching_results.tsv and candidate_pairs.tsv (tab-separated, unquoted).

    matches / candidates: long frames (s1_id, q_id). s1_ids: every test Source-1 id.
    Matches are forced to be a subset of candidates.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = matches.join(candidates.select("s1_id", "q_id").unique(), on=["s1_id", "q_id"], how="semi")
    for pairs, name, col in [
        (matches, "matching_results.tsv", "matched_entity_ids"),
        (candidates, "candidate_pairs.tsv", "candidate_entity_ids"),
    ]:
        _id_lists(pairs, s1_ids, col).write_csv(out_dir / name, separator="\t", quote_style="never")
    print(f"  wrote {out_dir / 'matching_results.tsv'} ({matches.height:,} matches, "
          f"{matches['s1_id'].n_unique():,} of {len(s1_ids):,} S1 entities matched)")


def test_s1_ids(data_dir=config.DATA_DIR):
    """Every Source-1 id of the test set (each needs exactly one output row)."""
    return pl.read_csv(config.source_path(data_dir, "test", "source1"), separator="\t",
                       infer_schema=False, columns=["entity_id"])["entity_id"]
