"""Stage 1b — fill in missing states from the city, with a city -> state table learned from S1.

35% of French S2/S3 addresses end with a city only ("19 rue de braouet, la teste de buch"),
so they had no state: they fell back to whole-country blocking and looked like the
"no state = empty address" records the model saw in training.

S1 addresses always carry "<city>, <state>". A digit-free segment next to the state segment
is taken as a city; a city maps to a state when it occurs >= MIN_COUNT times with >= MIN_SHARE
agreement within its country. Records without a state whose address contains a known city
get that state (addr_state_inferred = True). Learned only from the provided S1 files.

    python -m src.infer_state        (after src.normalize, before src.block)
"""
import argparse

import polars as pl

from . import config

MIN_COUNT, MIN_SHARE = 3, 0.9


def _segments(df):
    """Long frame: one row per address segment with its position."""
    return (
        df.select("r", "country", seg=pl.col("addr_norm").str.split(", "))
        .explode("seg", empty_as_null=True)
        .drop_nulls("seg")
        .with_columns(pos=pl.int_range(pl.len()).over("r"))
    )


def learn_city_map(work_dir):
    """(country, city) -> state from S1 records of both splits."""
    s1 = pl.concat([
        pl.read_parquet(config.norm_path(work_dir, split, "source1"), columns=["country", "addr_norm", "addr_state"])
        for split in config.SPLITS
    ]).drop_nulls("addr_state").with_row_index("r")
    segs = _segments(s1)
    state_pos = segs.join(s1.select("r", "addr_state"), on="r").filter(pl.col("seg") == pl.col("addr_state"))
    cities = (
        segs.join(state_pos.select("r", spos="pos", state="addr_state"), on="r")
        .filter(((pl.col("pos") - pl.col("spos")).abs() == 1), ~pl.col("seg").str.contains(r"\d"),
                pl.col("seg").str.len_chars() >= 3, pl.col("seg") != pl.col("state"))
        .group_by("country", "seg", "state").len()
        .with_columns(total=pl.col("len").sum().over("country", "seg"))
        .sort("len", descending=True).group_by("country", "seg").first()
        .filter(pl.col("len") >= MIN_COUNT, pl.col("len") / pl.col("total") >= MIN_SHARE)
    )
    return cities.select("country", "seg", inferred="state")


def apply_city_map(df, city_map):
    """Fill addr_state from a known city segment where it is missing."""
    df = df.drop("addr_state_inferred", strict=False).with_row_index("r")
    missing = df.filter(pl.col("addr_state").is_null())
    found = (
        _segments(missing).join(city_map, on=["country", "seg"])
        .sort("pos").group_by("r").agg(pl.col("inferred").first())
    )
    return (
        df.join(found, on="r", how="left")
        .with_columns(addr_state_inferred=pl.col("inferred").is_not_null(),
                      addr_state=pl.coalesce("addr_state", "inferred"))
        .drop("r", "inferred")
    )


def main():
    parser = argparse.ArgumentParser(description="Infer missing states from city names")
    parser.add_argument("--work-dir", default=config.WORK_DIR)
    args = parser.parse_args()

    city_map = learn_city_map(args.work_dir)
    print(f"  city -> state table: {city_map.height:,} cities "
          f"({city_map.group_by('country').len().sort('country').rows()})")
    for split in config.SPLITS:
        for source in config.SOURCES:
            path = config.norm_path(args.work_dir, split, source)
            df = apply_city_map(pl.read_parquet(path), city_map)
            df.write_parquet(path)
            stats = df.group_by("country").agg(
                no_state=pl.col("addr_state").is_null().mean().round(3),
                inferred=pl.col("addr_state_inferred").mean().round(3),
            ).sort("country")
            print(f"  {split}_{source}: {stats.rows()}")


if __name__ == "__main__":
    main()
