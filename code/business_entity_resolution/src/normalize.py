"""Stage 1 — normalise business names and addresses for every source file.

Run from code/business_entity_resolution:
    python -m src.normalize [--data-dir DIR] [--work-dir DIR] [--limit N]

Writes <work>/translit_dict.json and <work>/norm/{split}_{source}.parquet with columns:
    entity_id, country, name_raw, addr_raw,
    name_norm     cleaned full name (legal forms canonicalised, e.g. "ram marketing pvt ltd")
    name_core     name without legal forms / honorifics / dba marker ("ram marketing")
    name_legal    sorted canonical legal forms ("ltd pvt")
    name_nospace  name_core without spaces (matches domain-style names)
    name_dba      name contained a d/b/a style separator
    name_indic    raw name contained Indic script
    addr_norm     cleaned address, comma-separated segments, abbreviations expanded
    addr_state    canonical state/region if any segment is one (US code, India/France name)
    addr_nums     distinct numbers in the address, leading zeros stripped
    addr_house    first number (usually the house/plot number)
    addr_indic    raw address contained Indic script
One file is processed at a time, so peak memory stays a few GB (fits Colab).
"""
import argparse
import time
from pathlib import Path

import polars as pl

from . import config
from .text_maps import (
    ADDRESS_PLACEHOLDER_PATTERN,
    DBA_PATTERN,
    DOMAIN_PATTERN,
    LEGAL_TOKENS,
    NAME_NOISE_TOKENS,
    NAME_TOKEN_MAP,
    address_abbrev_for,
    ambiguous_states_for,
    state_map_for,
)
from .translit import INDIC_CHAR_PL, Transliterator, mine_dictionary


def read_source(path, limit=None):
    """Read one challenge TSV with every column as a string."""
    return pl.read_csv(path, separator="\t", infer_schema=False, n_rows=limit)


def clean_text(expr):
    """Lower-case and fold accents (é -> e, œ -> oe)."""
    return (
        expr.fill_null("").str.to_lowercase()
        .str.replace_all("œ", "oe").str.replace_all("æ", "ae")
        .str.normalize("NFKD").str.replace_all(r"\p{Mn}", "")
    )


def _dedupe_adjacent(tokens):
    """Drop a token equal to the one before it ("sodyne sodyne" -> "sodyne")."""
    return tokens.list.eval(pl.element().filter(pl.element().ne_missing(pl.element().shift(1))))


def normalize_names(df, src_col):
    """Add the name_* columns computed from the (transliterated) column `src_col`."""
    text = (
        clean_text(pl.col(src_col))
        .str.replace_all(r"\(\s*id\s*[:#-]?\s*\d+\s*\)|\bid\s*:\s*\d+", " ")
        .str.replace_all(r"\b\d{6,}\b", " ")  # injected ids / phone numbers ("summit 1778648767")
        .str.replace_all(r"\((?:india|france|usa|us)\)", " ")
        .str.replace_all(r"\bm\s*/\s*s\b", " ")
        .str.replace_all(DBA_PATTERN, " dba ")
        .str.replace_all(r"www\.|^@", "")
        .str.replace_all(DOMAIN_PATTERN, " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"\b(\p{L})\.", "${1}")  # s.a.r.l. -> sarl, l.l.c. -> llc
        .str.replace_all(r"['’`]", "")
        .str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.strip_chars()
    )
    tokens = text.str.split(" ").list.eval(pl.element().replace(NAME_TOKEN_MAP))
    df = df.with_columns(_tok=tokens.list.eval(pl.element().filter(pl.element() != "")))

    drop = sorted(LEGAL_TOKENS | NAME_NOISE_TOKENS)
    core = _dedupe_adjacent(pl.col("_tok").list.eval(pl.element().filter(~pl.element().is_in(drop))))
    norm = pl.col("_tok").list.join(" ")
    df = df.with_columns(
        name_norm=norm,
        name_core=pl.when(core.list.len() > 0).then(core.list.join(" ")).otherwise(norm),
        name_legal=pl.col("_tok").list.eval(pl.element().filter(pl.element().is_in(sorted(LEGAL_TOKENS))))
        .list.unique().list.sort().list.join(" "),
        name_dba=pl.col("_tok").list.contains("dba"),
    )
    return df.with_columns(name_nospace=pl.col("name_core").str.replace_all(" ", "")).drop("_tok")


def normalize_addresses(df, src_col, country):
    """Add the addr_* columns for rows of a single country."""
    abbrev = address_abbrev_for(country)
    states = state_map_for(country)
    state_expr = (
        pl.col("seg").replace_strict(states, default=None, return_dtype=pl.String)
        if states else pl.lit(None, dtype=pl.String)
    )

    text = (
        clean_text(pl.col(src_col))
        .str.replace_all(ADDRESS_PLACEHOLDER_PATTERN, " ")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\p{L}\p{N},]+", " ")
    )
    # Long format: one row per comma segment, so states can be matched on whole segments.
    seg = (
        df.select("entity_id", seg=text.str.split(","))
        .explode("seg", empty_as_null=True)
        .with_columns(pl.col("seg").str.replace_all(r"\s+", " ").str.strip_chars())
        .filter(pl.col("seg") != "")
        .with_columns(
            state=state_expr,
            _weak=pl.col("seg").is_in(ambiguous_states_for(country)),
            _pos=pl.int_range(pl.len()).over("entity_id"),
        )
        .with_columns(
            seg=pl.coalesce(
                "state",
                pl.col("seg").str.split(" ")
                .list.eval(pl.element().replace(abbrev))
                .list.eval(pl.element().filter(pl.element() != ""))
                .list.join(" "),
            )
        )
        .filter(pl.col("seg") != "")
    )
    per_row = seg.group_by("entity_id", maintain_order=True).agg(
        addr_norm=pl.col("seg").str.join(", "),
        # Several candidates ("Oregon, Ohio"): prefer unambiguous ones, then the last segment.
        addr_state=pl.col("state").sort_by("_weak", "_pos", descending=[True, False]).drop_nulls().last(),
    )
    per_row = per_row.with_columns(
        addr_nums=pl.col("addr_norm").str.extract_all(r"\d+")
        .list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique(maintain_order=True),
    ).with_columns(addr_house=pl.col("addr_nums").list.first())

    return df.join(per_row, on="entity_id", how="left").with_columns(
        pl.col("addr_norm").fill_null(""),
        pl.col("addr_nums").fill_null(pl.lit([], dtype=pl.List(pl.String))),
    )


def normalize_frame(df, translit):
    """Full normalisation of one raw source frame."""
    df = df.rename({"business_name": "name_raw", "business_address": "addr_raw"})
    df = translit.column(df, "name_raw", "_name")
    df = translit.column(df, "addr_raw", "_addr")
    df = df.with_columns(
        name_indic=pl.col("name_raw").str.contains(INDIC_CHAR_PL),
        addr_indic=pl.col("addr_raw").fill_null("").str.contains(INDIC_CHAR_PL),
    )
    df = normalize_names(df, "_name")
    # Country-specific address rules: process each country label separately (open set).
    parts = [
        normalize_addresses(part, "_addr", part["country"][0])
        for part in df.partition_by("country", maintain_order=True)
    ]
    return pl.concat(parts).drop("_name", "_addr")


def build_transliterator(data_dir, work_dir, limit=None):
    """Mine the Indic -> Latin dictionary from the training data and save it."""
    path = Path(work_dir) / "translit_dict.json"
    s1 = read_source(config.source_path(data_dir, "train", "source1"), limit)
    others = pl.concat([
        read_source(config.source_path(data_dir, "train", s), limit) for s in ("source2", "source3")
    ])
    gt = read_source(Path(data_dir) / "train" / "train_ground_truth.tsv", limit)
    translit = Transliterator(mine_dictionary(s1, others, gt))
    translit.save(path)
    print(f"  transliteration dictionary: {len(translit.mapping):,} tokens -> {path}")
    return translit


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    parser.add_argument("--limit", type=int, default=None, help="rows per file (quick test runs)")
    args = parser.parse_args()

    (Path(args.work_dir) / "norm").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    translit = build_transliterator(args.data_dir, args.work_dir, args.limit)
    print(f"  mined in {time.time() - t0:.0f}s")

    for split in config.SPLITS:
        for source in config.SOURCES:
            t = time.time()
            raw = read_source(config.source_path(args.data_dir, split, source), args.limit)
            out = normalize_frame(raw, translit)
            assert out.height == raw.height, "row count changed during normalisation"
            out.write_parquet(config.norm_path(args.work_dir, split, source))
            print(f"  {split}_{source}: {out.height:,} rows in {time.time() - t:.0f}s")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
