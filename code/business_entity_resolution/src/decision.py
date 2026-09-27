"""Turn pair scores into matches: each S2/S3 record goes to at most one S1 (its best)."""
import polars as pl


def best_per_query(scored, score_col="p"):
    """Per query: best S1, its score, and the margin over the runner-up (0 if no runner-up)."""
    s = pl.col(score_col)
    return scored.group_by("q_id").agg(
        s1_id=pl.col("s1_id").sort_by(s, descending=True).first(),
        score=s.max(),
        second=pl.when(pl.len() > 1).then(s.top_k(2).min()).otherwise(0.0),
    ).with_columns(margin=pl.col("score") - pl.col("second"))


def select_matches(best, t, margin):
    """Accepted (s1_id, q_id) pairs."""
    return best.filter(pl.col("score") >= t, pl.col("margin") >= margin).select("s1_id", "q_id")
