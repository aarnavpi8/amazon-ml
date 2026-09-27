"""Stage 3 — features for (S2/S3 query, S1 candidate) pairs.

Four groups, all country-agnostic (country itself is never a feature, so France is scored
with the same model):
  block    cosines and pass flags from blocking
  context  rank / gap of the pair inside its query's and its S1's candidate lists
  string   rapidfuzz similarities of cleaned names and addresses
  struct   legal-form, house-number, number-set and state agreement, acronym, flags
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import config
from .text_maps import BLOCK_STATE_ALIASES

ATTRS = ["entity_id", "name_core", "name_norm", "name_legal", "name_nospace",
         "addr_norm", "addr_state", "addr_house", "addr_nums"]
STATE_ALIAS = {k: v for m in BLOCK_STATE_ALIASES.values() for k, v in m.items()}

BLOCK = ["name_cos", "addr_cos", "block_score", "by_name", "by_addr", "by_exact"]
CONTEXT = ["q_rank", "q_n", "q_gap", "q_name_rank", "q_addr_rank", "s_rank", "s_n", "s_gap"]
STRING = ["n_ratio", "n_partial", "n_tset", "n_tsort", "n_jw", "nf_tset", "n_leet_ratio",
          "a_ratio", "a_tset", "a_street_tset", "house_lev"]
STRUCT = ["nospace_eq", "acronym", "len1", "len2", "legal_eq", "legal_conflict", "legal_missing",
          "house_eq", "house_conflict", "house_missing", "house_prefix", "house_suffix", "house_reldiff",
          "nums_common", "nums_jacc", "state_eq", "state_conflict", "q_addr_empty", "q_indic", "q_dba", "q_s3"]
# Look-alike digits injected into names ("8aba" -> "baba", "0campo" -> "ocampo").
LEET_FROM, LEET_TO = ["0", "1", "3", "5", "8"], ["o", "l", "e", "s", "b"]
FEATURES = BLOCK + CONTEXT + STRING + STRUCT


def load_attrs(work_dir, split):
    """Normalised attributes keyed by s1_id (suffix _1) and q_id (suffix _2)."""
    s1 = pl.read_parquet(config.norm_path(work_dir, split, "source1"), columns=ATTRS)
    q = pl.concat([
        pl.read_parquet(config.norm_path(work_dir, split, s), columns=ATTRS + ["name_indic", "name_dba"])
        for s in ("source2", "source3")
    ])
    s1 = s1.rename({c: f"{c}_1" for c in ATTRS[1:]} | {"entity_id": "s1_id"})
    q = q.rename({c: f"{c}_2" for c in ATTRS[1:] + ["name_indic", "name_dba"]} | {"entity_id": "q_id"})
    return s1, q


def add_context(cands):
    """Rank / gap of each pair within its query's and its S1's candidate lists."""
    bs = pl.col("block_score")
    rank = lambda col, key: pl.col(col).rank("ordinal", descending=True).over(key)
    return cands.with_columns(
        q_rank=rank("block_score", "q_id"),
        q_n=pl.len().over("q_id"),
        q_gap=bs.max().over("q_id") - bs,
        q_name_rank=rank("name_cos", "q_id"),
        q_addr_rank=rank("addr_cos", "q_id"),
        s_rank=rank("block_score", "s1_id"),
        s_n=pl.len().over("s1_id"),
        s_gap=bs.max().over("s1_id") - bs,
    )


def _pairwise(a, b, scorer):
    """Element-wise similarity of two equal-length string lists (all cores)."""
    return process.cpdist(a, b, scorer=scorer, processor=None, workers=-1).astype(np.float32)


def string_features(df):
    """rapidfuzz similarities; texts are already normalised, so no processor is needed."""
    get = lambda c: df[c].fill_null("").to_list()
    get_expr = lambda e: df.select(e.fill_null(""))[:, 0].to_list()
    leet = lambda c: pl.col(c).str.replace_many(LEET_FROM, LEET_TO)
    street = lambda c: pl.col(c).str.replace_all(r"\d+", " ")
    n1, n2 = get("name_core_1"), get("name_core_2")
    f1, f2 = get("name_norm_1"), get("name_norm_2")
    a1, a2 = get("addr_norm_1"), get("addr_norm_2")
    h1, h2 = get("addr_house_1"), get("addr_house_2")
    house_missing = (df["addr_house_1"].is_null() | df["addr_house_2"].is_null()).to_numpy()
    house_lev = _pairwise(h1, h2, Levenshtein.distance)
    house_lev[house_missing] = np.nan
    return df.with_columns(
        n_ratio=_pairwise(n1, n2, fuzz.ratio),
        n_partial=_pairwise(n1, n2, fuzz.partial_ratio),
        n_tset=_pairwise(n1, n2, fuzz.token_set_ratio),
        n_tsort=_pairwise(n1, n2, fuzz.token_sort_ratio),
        n_jw=_pairwise(n1, n2, JaroWinkler.normalized_similarity),
        nf_tset=_pairwise(f1, f2, fuzz.token_set_ratio),
        n_leet_ratio=_pairwise(get_expr(leet("name_core_1")), get_expr(leet("name_core_2")), fuzz.ratio),
        a_ratio=_pairwise(a1, a2, fuzz.ratio),
        a_tset=_pairwise(a1, a2, fuzz.token_set_ratio),
        # Same street with a different number is the typical decoy; compare streets without numbers.
        a_street_tset=_pairwise(get_expr(street("addr_norm_1")), get_expr(street("addr_norm_2")), fuzz.token_set_ratio),
        house_lev=house_lev,
    )


def struct_features(df):
    """Agreement / conflict flags computed with polars expressions."""
    col = pl.col
    both = lambda a, b: col(a).is_not_null() & col(b).is_not_null()
    initials = lambda c: col(c).str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("")
    multi_word = lambda c: col(c).str.contains(" ")
    st1, st2 = col("addr_state_1").replace(STATE_ALIAS), col("addr_state_2").replace(STATE_ALIAS)
    legal1, legal2 = col("name_legal_1").fill_null(""), col("name_legal_2").fill_null("")
    legal_common = legal1.str.split(" ").list.set_intersection(legal2.str.split(" ")).list.len()
    nums_common = col("addr_nums_1").list.set_intersection("addr_nums_2").list.len()
    nums_union = col("addr_nums_1").list.set_union("addr_nums_2").list.len()
    return df.with_columns(
        nospace_eq=col("name_nospace_1") == col("name_nospace_2"),
        acronym=(multi_word("name_core_1") & (initials("name_core_1") == col("name_nospace_2")))
        | (multi_word("name_core_2") & (initials("name_core_2") == col("name_nospace_1"))),
        len1=col("name_core_1").str.len_chars(),
        len2=col("name_core_2").str.len_chars(),
        legal_eq=(legal1 != "") & (legal1 == legal2),
        legal_conflict=(legal1 != "") & (legal2 != "") & (legal_common == 0),
        legal_missing=(legal1 == "") != (legal2 == ""),
        house_eq=(both("addr_house_1", "addr_house_2") & (col("addr_house_1") == col("addr_house_2"))).fill_null(False),
        house_conflict=(both("addr_house_1", "addr_house_2") & (col("addr_house_1") != col("addr_house_2"))).fill_null(False),
        house_missing=col("addr_house_1").is_null() | col("addr_house_2").is_null(),
        # Noise drops a digit ("118" -> "11", "10108" -> "108"); decoys change one ("7265" -> "7269").
        house_prefix=(both("addr_house_1", "addr_house_2") & (col("addr_house_1") != col("addr_house_2"))
                      & (col("addr_house_1").str.starts_with(col("addr_house_2"))
                         | col("addr_house_2").str.starts_with(col("addr_house_1")))).fill_null(False),
        house_suffix=(both("addr_house_1", "addr_house_2") & (col("addr_house_1") != col("addr_house_2"))
                      & (col("addr_house_1").str.ends_with(col("addr_house_2"))
                         | col("addr_house_2").str.ends_with(col("addr_house_1")))).fill_null(False),
        house_reldiff=(
            (col("addr_house_1").cast(pl.Float64, strict=False) - col("addr_house_2").cast(pl.Float64, strict=False)).abs()
            / pl.max_horizontal(col("addr_house_1").cast(pl.Float64, strict=False),
                                col("addr_house_2").cast(pl.Float64, strict=False), pl.lit(1.0))
        ),
        nums_common=nums_common,
        nums_jacc=pl.when(nums_union > 0).then(nums_common / nums_union).otherwise(0.0),
        state_eq=(st1 == st2).fill_null(False),
        state_conflict=(st1.is_not_null() & st2.is_not_null() & (st1 != st2)).fill_null(False),
        q_addr_empty=col("addr_norm_2").fill_null("") == "",
        q_indic=col("name_indic_2"),
        q_dba=col("name_dba_2"),
        q_s3=col("q_id").str.starts_with("S3-"),
    )


def build_features(cands, s1_attr, q_attr, chunk=3_000_000):
    """cands (with context columns) -> (q_id, s1_id, *FEATURES as Float32), processed in chunks."""
    out = []
    for start in range(0, cands.height, chunk):
        part = (
            cands.slice(start, chunk)
            .join(s1_attr, on="s1_id", how="left")
            .join(q_attr, on="q_id", how="left")
        )
        part = struct_features(string_features(part))
        out.append(part.select("q_id", "s1_id", *[pl.col(f).cast(pl.Float32) for f in FEATURES]))
    return pl.concat(out)
