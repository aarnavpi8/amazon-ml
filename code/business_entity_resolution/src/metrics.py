"""Evaluation helpers: ground-truth loading and the challenge's macro F0.5."""
from pathlib import Path

import polars as pl


def load_truth_pairs(data_dir):
    """Ground truth as long pairs (s1_id, q_id) plus the list of all train S1 ids."""
    gt = pl.read_csv(Path(data_dir) / "train" / "train_ground_truth.tsv", separator="\t", infer_schema=False)
    pairs = (
        gt.select(s1_id="source1_entity_id", q_id=pl.col("matched_entity_ids").str.split(","))
        .explode("q_id", empty_as_null=True)
        .drop_nulls()
    )
    return pairs, gt["source1_entity_id"]


def macro_f05(pred, truth, s1_ids, beta=0.5):
    """Challenge metric: F-beta per S1 entity, averaged over all S1 entities.

    pred / truth: long frames with columns (s1_id, q_id). s1_ids: every S1 entity evaluated.
    An entity with no true matches scores 1.0 for an empty prediction and 0.0 otherwise.
    """
    b2 = beta * beta
    tp = pred.join(truth, on=["s1_id", "q_id"]).group_by("s1_id").len("tp")
    n_pred = pred.group_by("s1_id").len("n_pred")
    n_true = truth.group_by("s1_id").len("n_true")
    per = (
        pl.DataFrame({"s1_id": s1_ids})
        .join(tp, on="s1_id", how="left")
        .join(n_pred, on="s1_id", how="left")
        .join(n_true, on="s1_id", how="left")
        .fill_null(0)
        .with_columns(
            p=pl.col("tp") / pl.col("n_pred"),
            r=pl.col("tp") / pl.col("n_true"),
        )
        .with_columns(
            f=pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
            .when(pl.col("tp") == 0).then(0.0)
            .otherwise((1 + b2) * pl.col("p") * pl.col("r") / (b2 * pl.col("p") + pl.col("r")))
        )
    )
    return per["f"].mean()
