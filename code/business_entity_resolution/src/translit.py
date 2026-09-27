"""Indic-script -> Latin transliteration.

About 18% of Indian training matches pair a Latin-script Source-1 name with an S2/S3 name
written in Devanagari, Tamil, Bengali, ... Two layers turn those into comparable Latin text:

1. A token dictionary mined from the training ground truth: when a Latin S1 name and its
   matched Indic S2/S3 name have the same number of tokens, the tokens are aligned by
   position and the most frequent Latin spelling per Indic token is kept. The same is done
   for Indic state names in addresses against the S1 state segment.
2. A rule-based fallback (indic_transliteration, MIT licence) for tokens the dictionary
   has not seen, simplified to plain ASCII with a light schwa-deletion heuristic.
"""
import json
import re
import unicodedata
from functools import lru_cache

import polars as pl
from indic_transliteration import sanscript

# Unicode blocks Devanagari (U+0900) .. Malayalam (U+0D7F), plus zero-width (non-)joiners.
INDIC_TOKEN_RE = re.compile(r"[ऀ-ൿ‌‍]+")
INDIC_CHAR_PL = r"[\x{0900}-\x{0D7F}]"
INDIC_TOKEN_PL = r"^[\x{0900}-\x{0D7F}\x{200c}\x{200d}]+$"
WORD_PL = r"[\p{L}\p{M}\p{N}\x{200c}\x{200d}]+"

# One 128-code-point block per script, in Unicode order starting at U+0900.
_SCHEMES = [
    sanscript.DEVANAGARI, sanscript.BENGALI, sanscript.GURMUKHI, sanscript.GUJARATI,
    sanscript.ORIYA, sanscript.TAMIL, sanscript.TELUGU, sanscript.KANNADA, sanscript.MALAYALAM,
]
_INDIC_DIGITS = {
    cp: str(unicodedata.digit(chr(cp)))
    for cp in range(0x0900, 0x0D80)
    if unicodedata.digit(chr(cp), None) is not None
}


def _strip_accents(text):
    """Drop combining marks after NFKD decomposition (ā -> a, ṭ -> t)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=None)
def rule_transliterate(token):
    """Rule-based romanisation of one Indic token, reduced to [a-z0-9]."""
    first = next((c for c in token if 0x0900 <= ord(c) < 0x0D80), None)
    if first is None:
        return ""
    scheme = _SCHEMES[(ord(first) - 0x0900) // 0x80]
    out = sanscript.transliterate(token.translate(_INDIC_DIGITS), scheme, sanscript.IAST)
    out = out.replace("ṃ", "n").replace("ṁ", "n")
    out = re.sub(r"[^a-z0-9]", "", _strip_accents(out).lower())
    # Schwa deletion: "rama" -> "ram", "limiteda" -> "limited".
    if len(out) > 3 and out[-1] == "a" and out[-2] not in "aeiou":
        out = out[:-1]
    return out


class Transliterator:
    """Replaces every Indic token in a string with its Latin form."""

    def __init__(self, mapping):
        self.mapping = mapping

    def token(self, tok):
        """Latin form of one Indic token: mined dictionary first, rules second."""
        return self.mapping.get(tok) or rule_transliterate(tok)

    def text(self, text):
        """Transliterate all Indic runs inside a string, leaving other text untouched."""
        return INDIC_TOKEN_RE.sub(lambda m: self.token(m.group()), text)

    def column(self, df, col, out):
        """Add column `out` = `col` with Indic tokens transliterated (only Indic rows touched)."""
        uniq = df.filter(pl.col(col).str.contains(INDIC_CHAR_PL))[col].unique().to_list()
        mapped = pl.col(col).replace(uniq, [self.text(u) for u in uniq]) if uniq else pl.col(col)
        return df.with_columns(mapped.alias(out))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, ensure_ascii=False, indent=0, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def _latin_tokens(col):
    """Accent-free lower-case [a-z0-9] tokens of a Latin-script column."""
    return (
        pl.col(col).fill_null("").str.to_lowercase().str.normalize("NFKD")
        .str.replace_all(r"\p{Mn}", "").str.extract_all(r"[a-z0-9]+")
    )


def _aligned_counts(df, latin, indic):
    """Count (indic_token, latin_text) pairs from two token lists.

    Equal-length lists are aligned position by position. A single Indic token facing
    several Latin tokens maps to the whole phrase (தமிழ்நாடு -> "tamil nadu").
    """
    toks = df.select(t1=latin, t2=pl.col(indic).str.to_lowercase().str.extract_all(WORD_PL)).filter(
        pl.col("t1").list.len() > 0, pl.col("t2").list.len() > 0
    )
    same = toks.filter(pl.col("t1").list.len() == pl.col("t2").list.len()).explode("t1", "t2")
    phrase = toks.filter(pl.col("t2").list.len() == 1, pl.col("t1").list.len() > 1).select(
        t1=pl.col("t1").list.join(" "), t2=pl.col("t2").list.first()
    )
    return (
        pl.concat([same, phrase])
        .filter(pl.col("t2").str.contains(INDIC_TOKEN_PL))
        .group_by("t2", "t1")
        .len()
    )


def mine_dictionary(s1, others, ground_truth, min_count=2, min_share=0.5):
    """Mine an Indic-token -> Latin-token dictionary from matched training pairs.

    s1 / others: raw source frames (entity_id, business_name, business_address).
    ground_truth: raw train_ground_truth frame.
    """
    indic_others = others.filter(
        pl.col("business_name").str.contains(INDIC_CHAR_PL)
        | pl.col("business_address").fill_null("").str.contains(INDIC_CHAR_PL)
    ).select(mid="entity_id", name2="business_name", addr2="business_address")

    pairs = (
        ground_truth.select(s1_id="source1_entity_id", mid=pl.col("matched_entity_ids").str.split(","))
        .explode("mid", empty_as_null=True)
        .join(indic_others, on="mid")
        .join(s1.select(s1_id="entity_id", name1="business_name", addr1="business_address"), on="s1_id")
    )

    # Names: whole name vs whole name.
    names = _aligned_counts(pairs, _latin_tokens("name1"), "name2")

    # Addresses: the single Indic segment of addr2 (usually the state) vs the S1 last segment.
    seg = pairs.select(
        s1_state=pl.col("addr1").str.split(",").list.last().str.strip_chars(),
        indic_seg=pl.col("addr2").fill_null("").str.split(",")
        .list.eval(pl.element().filter(pl.element().str.contains(INDIC_CHAR_PL)).str.strip_chars()),
    ).filter(pl.col("indic_seg").list.len() == 1).with_columns(pl.col("indic_seg").list.first())
    states = _aligned_counts(seg, _latin_tokens("s1_state"), "indic_seg")

    counts = pl.concat([names, states]).group_by("t2", "t1").agg(pl.col("len").sum())
    best = (
        counts.with_columns(total=pl.col("len").sum().over("t2"))
        .sort("len", descending=True)
        .group_by("t2")
        .first()
        .filter(pl.col("len") >= min_count, pl.col("len") / pl.col("total") >= min_share)
    )
    return dict(zip(best["t2"].to_list(), best["t1"].to_list()))
