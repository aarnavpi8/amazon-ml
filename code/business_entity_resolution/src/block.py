"""Stage 2 — candidate generation (blocking).

Every S2/S3 record ("query") retrieves its most similar S1 records inside its own country:
  name pass   TF-IDF char 3-grams of name_core, within each state block
  addr pass   TF-IDF char 3-grams of the address without state segments, same blocks
  exact pass  same name_nospace, or same "house-number street-word" key, whole country
Each pass retrieves its top RETRIEVE_TOPN by its own cosine, then re-ranks them by
name_cos + addr_cos and keeps PASS_KEEP (so identical generic names are separated by address).
A query mentioning several states searches each of those blocks; a query with no state
searches the whole country. The union of all passes is capped to KEEP_PER_QUERY per query
(the final cut to K happens in `cap_candidates`).

Run from code/business_entity_resolution:
    python -m src.block --split train
    python -m src.block --split test
Writes <work>/cand/{split}/{country}.parquet (resumable: finished countries are skipped).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config
from .text_maps import BLOCK_STATE_ALIASES, STATE_MAPS

RETRIEVE_TOPN = 20       # per pass and block, by that pass's own cosine
NOSTATE_TOPN = 50        # queries without a state search the whole country, so fetch more
PASS_KEEP = 6            # kept per query and pass after re-ranking by name_cos + addr_cos
NAME_MIN_COS, ADDR_MIN_COS = 0.3, 0.3
MAX_DF = 0.02            # drop char 3-grams present in >2% of S1 records (4x faster)
EXACT_MAX_S1 = 3         # exact keys shared by more S1 records than this are too generic
KEEP_PER_QUERY = 15      # stored per query; the model-facing K is chosen later
COLS = ["entity_id", "country", "name_core", "name_nospace", "addr_norm"]


def load_split(work_dir, split):
    """Normalised S1 and S2+S3 (queries) for one split."""
    s1 = pl.read_parquet(config.norm_path(work_dir, split, "source1"), columns=COLS)
    q = pl.concat([
        pl.read_parquet(config.norm_path(work_dir, split, s), columns=COLS + ["name_indic"])
        for s in ("source2", "source3")
    ])
    return s1, q


def prepare(df, country):
    """Add state list, state-free address text and the exact address key; index rows 0..n-1."""
    canon = sorted(set(STATE_MAPS.get(country, {}).values()))
    aliases = BLOCK_STATE_ALIASES.get(country, {})
    segs = pl.col("addr_norm").str.split(", ")
    return df.with_row_index("idx").with_columns(
        states=segs.list.eval(pl.element().filter(pl.element().is_in(canon)))
        .list.eval(pl.element().replace(aliases)).list.unique(),
        addr_text=segs.list.eval(pl.element().filter(~pl.element().is_in(canon))).list.join(", "),
    ).with_columns(
        addr_key=pl.col("addr_text").str.extract(r"(\d+ [a-z]{3,})", 1),
        name_key=pl.when(pl.col("name_nospace").str.len_chars() >= 4).then(pl.col("name_nospace")),
    )


def tfidf(s1_text, q_text):
    """Fit char 3-gram TF-IDF on S1 texts; return L2-normalised (S1, query) matrices or None."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                          dtype=np.float32, max_df=MAX_DF)
    try:
        S = vec.fit_transform(s1_text)
    except ValueError:  # empty vocabulary (e.g. all addresses empty)
        return None, None
    return S.tocsr(), vec.transform(q_text).tocsr()


def _combined_score(mats, qi, si):
    """name_cos + addr_cos for pairs (qi, si); mats = (Q_name, S_name, Q_addr, S_addr)."""
    Qn, Sn, Qa, Sa = mats
    return rowwise_cos(Qn, Sn, qi, si) + rowwise_cos(Qa, Sa, qi, si)


def _search(Q, S, mats, q_idx, s_idx, top_n, threshold):
    """Retrieve top_n by this pass's cosine, re-rank by name+addr, keep PASS_KEEP per query."""
    if len(q_idx) == 0 or len(s_idx) == 0:
        return None
    C = sp_matmul_topn(Q[q_idx], S[s_idx].T.tocsr(), top_n=top_n, threshold=threshold, n_threads=-1).tocoo()
    qi, si = q_idx[C.row], s_idx[C.col]
    return (
        pl.DataFrame({"qi": qi.astype(np.uint32), "si": si.astype(np.uint32),
                      "score": _combined_score(mats, qi, si)})
        .sort("score", descending=True).group_by("qi").head(PASS_KEEP)
    )


def blocked_search(Q, S, mats, q, s1, threshold):
    """One retrieval pass inside state blocks; stateless queries search the whole country."""
    s_by_state = {
        st: np.asarray(ix) for st, ix in
        s1.select("idx", "states").explode("states", empty_as_null=True).drop_nulls("states")
        .group_by("states").agg("idx").iter_rows()
    }
    s_nostate = s1.filter(pl.col("states").list.len() == 0)["idx"].to_numpy()
    q_blocks = q.select("idx", "states").explode("states", empty_as_null=True)

    parts = []
    for st, qi in q_blocks.drop_nulls("states").group_by("states").agg("idx").iter_rows():
        s_idx = np.concatenate([s_by_state.get(st, np.empty(0, np.uint32)), s_nostate]).astype(np.int64)
        parts.append(_search(Q, S, mats, np.asarray(qi, np.int64), s_idx, RETRIEVE_TOPN, threshold))
    q_nostate = q_blocks.filter(pl.col("states").is_null())["idx"].to_numpy().astype(np.int64)
    parts.append(_search(Q, S, mats, q_nostate, np.arange(S.shape[0]), NOSTATE_TOPN, threshold))

    parts = [p for p in parts if p is not None]
    if not parts:
        return pl.DataFrame(schema={"qi": pl.UInt32, "si": pl.UInt32})
    # Multi-state queries were searched in several blocks: keep their overall best.
    return (
        pl.concat(parts).unique(["qi", "si"])
        .sort("score", descending=True).group_by("qi").head(PASS_KEEP)
        .select("qi", "si")
    )


def exact_pairs(q, s1, key):
    """Pairs sharing an exact key, skipping keys held by more than EXACT_MAX_S1 S1 records."""
    s_keys = s1.select("idx", key).drop_nulls(key)
    s_keys = s_keys.filter(pl.len().over(key) <= EXACT_MAX_S1)
    return (
        q.select(qi="idx", k=key).drop_nulls("k")
        .join(s_keys.rename({"idx": "si", key: "k"}), on="k")
        .select(pl.col("qi").cast(pl.UInt32), pl.col("si").cast(pl.UInt32))
    )


def rowwise_cos(A, B, ai, bi, chunk=2_000_000):
    """Cosine of row A[ai[j]] with B[bi[j]] for every j (rows are L2-normalised)."""
    out = np.zeros(len(ai), np.float32)
    if A is None:
        return out
    for start in range(0, len(ai), chunk):
        a, b = A[ai[start:start + chunk]], B[bi[start:start + chunk]]
        out[start:start + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def block_country(s1, q, country):
    """All candidate pairs for one country, with cosines and pass flags."""
    s1, q = prepare(s1, country), prepare(q, country)
    S_name, Q_name = tfidf(s1["name_core"].to_list(), q["name_core"].to_list())
    S_addr, Q_addr = tfidf(s1["addr_text"].to_list(), q["addr_text"].to_list())

    mats = (Q_name, S_name, Q_addr, S_addr)
    passes = []
    if S_name is not None:
        passes.append(blocked_search(Q_name, S_name, mats, q, s1, NAME_MIN_COS).with_columns(src=pl.lit(1, pl.UInt8)))
    if S_addr is not None:
        passes.append(blocked_search(Q_addr, S_addr, mats, q, s1, ADDR_MIN_COS).with_columns(src=pl.lit(2, pl.UInt8)))
    for key in ("name_key", "addr_key"):
        passes.append(exact_pairs(q, s1, key).with_columns(src=pl.lit(4, pl.UInt8)))

    pairs = pl.concat(passes).group_by("qi", "si").agg(pl.col("src").unique().sum().cast(pl.UInt8))
    qi, si = pairs["qi"].to_numpy(), pairs["si"].to_numpy()
    pairs = pairs.with_columns(
        name_cos=rowwise_cos(Q_name, S_name, qi, si),
        addr_cos=rowwise_cos(Q_addr, S_addr, qi, si),
    )
    pairs = (
        pairs.with_columns(block_score=pl.col("name_cos") + pl.col("addr_cos"))
        .sort("block_score", descending=True).group_by("qi").head(KEEP_PER_QUERY)
    )
    return (
        pairs.join(q.select(qi=pl.col("idx").cast(pl.UInt32), q_id="entity_id"), on="qi")
        .join(s1.select(si=pl.col("idx").cast(pl.UInt32), s1_id="entity_id"), on="si")
        .select(
            "q_id", "s1_id", "name_cos", "addr_cos", "block_score",
            by_name=(pl.col("src") & 1) > 0, by_addr=(pl.col("src") & 2) > 0, by_exact=(pl.col("src") & 4) > 0,
        )
    )


def cap_candidates(cands, k):
    """Final candidate set: the best k S1 records per query by block_score."""
    return cands.sort("block_score", descending=True).group_by("q_id").head(k)


def load_candidates(work_dir, split):
    """All countries' stored candidates for a split."""
    return pl.read_parquet(Path(work_dir) / "cand" / split / "*.parquet")


def main():
    parser = argparse.ArgumentParser(description="Candidate generation (blocking)")
    parser.add_argument("--split", choices=config.SPLITS, required=True)
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--force", action="store_true", help="recompute finished countries")
    args = parser.parse_args()

    out_dir = Path(args.work_dir) / "cand" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s1_all, q_all = load_split(args.work_dir, args.split)
    for country in sorted(s1_all["country"].unique().to_list()):
        path = out_dir / f"{country}.parquet"
        if path.exists() and not args.force:
            print(f"  {country}: exists, skipping")
            continue
        t = time.time()
        s1 = s1_all.filter(pl.col("country") == country)
        q = q_all.filter(pl.col("country") == country)
        cands = block_country(s1, q, country)
        cands.write_parquet(path)
        print(f"  {country}: {s1.height:,} S1, {q.height:,} queries -> {cands.height:,} pairs in {time.time() - t:.0f}s")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
