"""Reusable EDA diagnostic helpers shared across notebooks: outlier
detection, correlation/cardinality summaries, and text/list-valued
field quality checks.
"""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from langdetect import LangDetectException, detect
from scipy import stats
from sklearn.preprocessing import MultiLabelBinarizer

HTML_RE = re.compile(r"<[^>]+>")


def iqr_outlier_count(s):
    """Counts values in `s` that fall outside the 1.5x-IQR Tukey fences.
    This is the standard skew-robust outlier rule, complementary to z-score."""
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def zscore_outlier_count(s, threshold=3):
    """Counts values in `s` whose z-score magnitude exceeds `threshold`,
    flagging outliers under a normal-distribution assumption."""
    s = s.dropna()
    if s.std(ddof=0) == 0 or len(s) == 0:
        return 0
    z = stats.zscore(s)
    return int((np.abs(z) > threshold).sum())


def high_corr_pairs(corr, threshold=0.85):
    """Extracts feature pairs from a correlation matrix whose absolute
    correlation exceeds `threshold`, surfacing multicollinearity risks."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                pairs.append((cols[i], cols[j], round(r, 3)))
    return pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"])


# DECISION POINT: embedding-dimension heuristic — currently Guo & Berkhahn
# (2016), min(100, cardinality // 2 + 1). If the methodology changes to a
# different heuristic (or a learned/tuned dimension), update here and in the
# "Entity-embedding dimension heuristic" markdown cell that cites it.
def entity_embedding_dim(cardinality):
    """Suggests an entity-embedding dimension for a categorical feature from
    its cardinality, using the Guo & Berkhahn (2016) heuristic."""
    return min(100, (cardinality // 2) + 1)


def empty_string_report(df, cols):
    """Reports, per column, how many values are null versus present-but-empty
    strings — two distinct missingness signatures that `isnull()` alone conflates."""
    rows = []
    for c in cols:
        as_str = df[c].astype("string")
        rows.append(
            {
                "column": c,
                "null_count": int(df[c].isnull().sum()),
                "empty_string_count": int((as_str.str.strip() == "").sum()),
            }
        )
    return pd.DataFrame(rows)


def text_field_report(df, col, label):
    """Prints and returns the null and empty-string counts for a single
    free-text column."""
    null_count = int(df[col].isnull().sum())
    empty_count = int((df[col].fillna("").astype(str).str.strip() == "").sum())
    print(f"{label} ({col}) — nulls: {null_count}, empty strings: {empty_count}")
    return null_count, empty_count


# DECISION POINT: embedding model choice — currently
# sentence-transformers/all-MiniLM-L6-v2 (512-token max sequence length).
# If the project switches to a different sentence-embedding model, update
# the model name here so token-length stats reflect the model actually used.
@lru_cache(maxsize=1)
def _get_tokenizer():
    # Import and initialization both deferred to first call (not a
    # module-level import/global) so importing this module doesn't pull in
    # transformers/torch -- and trigger a HuggingFace Hub download -- for
    # every notebook that imports it, even ones that never call
    # token_length_stats.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def token_length_stats(texts, max_len=512, sample_size=5000, random_state=42):
    """Estimates the token-length distribution of a text column under the
    all-MiniLM-L6-v2 tokenizer, from a random sample (tokenizing the full
    column would be prohibitively slow for the TMDB-sized dataset), and
    reports the percentage of sampled rows exceeding `max_len` tokens."""
    tokenizer = _get_tokenizer()
    texts = texts.dropna().astype(str)
    texts = texts[texts.str.strip() != ""]
    if len(texts) > sample_size:
        texts = texts.sample(sample_size, random_state=random_state)
    lengths = texts.apply(lambda t: len(tokenizer.encode(t, truncation=False)))
    pct_over = float((lengths > max_len).mean() * 100)
    return lengths, pct_over


# DECISION POINT: langdetect is a statistical n-gram method that degrades
# badly on very short strings and on text dominated by foreign proper nouns
# (e.g. cast-name lists, short titles) — spot-checking TMDB overview flags
# showed ~5/6 flagged rows were false positives for exactly this reason, not
# genuine non-English text. Currently unfiltered: every flag is treated the
# same regardless of confidence or text length.
# TODO: revisit with either (a) langdetect.detect_langs() + a confidence
# threshold (e.g. >0.9) instead of detect(), or (b) a minimum character-length
# gate that reports very short text as "too short to classify" rather than
# confidently (mis)labeling it, if false positives keep being a problem.
def flag_non_english(series, n=100, random_state=42):
    """Samples a text column and flags entries `langdetect` identifies as
    non-English, for manual spot-checking rather than automatic filtering."""
    sample = series.dropna().astype(str)
    sample = sample[sample.str.strip() != ""]
    sample = sample.sample(min(n, len(sample)), random_state=random_state)

    def safe_detect(t):
        try:
            return detect(t)
        except LangDetectException:
            return "unknown"

    langs = sample.apply(safe_detect)
    non_english = sample[langs != "en"]
    pct = float(len(non_english) / len(sample) * 100) if len(sample) else 0.0
    print(f"{len(non_english)} / {len(sample)} sampled rows ({pct:.1f}%) flagged as non-English")
    return non_english, pct


def html_contamination_pct(series):
    """Returns the percentage of a text column's values containing
    HTML/markup tags."""
    texts = series.dropna().astype(str)
    if len(texts) == 0:
        return 0.0
    contaminated = texts.str.contains(HTML_RE)
    return float(contaminated.mean() * 100)


def near_duplicate_report(df, col, id_col="id"):
    """Flags rows whose text, after case/whitespace normalization, exactly
    matches another row's — a simple first-pass near-duplicate check."""
    text = df[col].dropna().astype(str).str.lower().str.strip()
    text = text.str.replace(r"\s+", " ", regex=True)
    text = text[text != ""]  # exclude empty strings (missing, not duplicate)
    dupe_mask = text.duplicated(keep=False)
    n_dupe_rows = int(dupe_mask.sum())
    n_dupe_groups = int(text[dupe_mask].nunique())
    print(f"{n_dupe_rows} rows share duplicated normalized text across {n_dupe_groups} distinct text groups")

    display_cols = [id_col] + (["imdb_id"] if "imdb_id" in df.columns else []) + [col]
    dupe_rows = (
        df.loc[text[dupe_mask].index, display_cols]
        .assign(_normalized=text[dupe_mask])
        .sort_values("_normalized")  # group duplicates together for inspection
        .drop(columns="_normalized")
    )
    return dupe_rows, n_dupe_rows


def parse_list_field(value):
    """Tries JSON / Python-literal list parsing first, falling back to a
    plain comma split, since TMDB genres/keywords in this dataset are
    stored as comma-separated strings, not JSON."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except (ValueError, SyntaxError):
            continue
    return [item.strip() for item in text.split(",") if item.strip()]


def list_field_summary(df, list_col):
    """Summarizes a parsed list-valued column: vocabulary size, count of
    long-tail items (<10 occurrences), empty-list rate, and per-row list-length
    statistics."""
    lengths = df[list_col].apply(len)
    exploded = df[list_col].explode()
    freq = exploded.value_counts()
    return {
        "vocab_size": int(exploded.nunique()),
        "long_tail_items": int((freq < 10).sum()),
        "pct_empty_lists": float((lengths == 0).mean() * 100),
        "length_describe": lengths.describe(),
        "freq": freq,
    }


# DECISION POINT: multi-hot vs. pooled-embedding cutoff — currently a flat
# vocab_threshold=5000, and the byte estimate assumes 1 byte/cell (uint8
# multi-hot). If this feels too coarse or the multi-hot approach doesn't pan
# out, revisit this threshold and/or evaluate alternatives here.
# TODO: explore pooled SBERT-of-item-names (or hashing-trick / frequency-based
# top-K + "Other" bucketing) as alternatives if multi-hot proves infeasible
# or underperforms for genres/keywords/amenities.
def multihot_feasibility(vocab_size, n_rows, vocab_threshold=5000):
    """Estimates the memory cost of multi-hot encoding a list-valued column
    and warns when the vocabulary is too large for that to be practical."""
    total_bytes = vocab_size * n_rows * 1
    print(
        f"Multi-hot matrix estimate: {vocab_size} vocab x {n_rows} rows "
        f"= {total_bytes:,} bytes (~{total_bytes / 1e6:.1f} MB)"
    )
    if vocab_size > vocab_threshold:
        print(
            "WARNING: vocabulary exceeds 5000 items — multi-hot encoding is "
            "infeasible; use pooled SBERT-of-item-names embeddings instead."
        )
    else:
        print("Multi-hot encoding is feasible at this vocabulary size.")


def top_item_cooccurrence(list_series, top_n=20):
    """Computes a co-occurrence count matrix for the top-N most frequent
    items in a list-valued column, to eyeball natural item clusters."""
    freq = list_series.explode().value_counts()
    top_items = freq.head(top_n).index.tolist()
    restricted = list_series.apply(lambda items: [i for i in items if i in top_items])
    mlb = MultiLabelBinarizer(classes=top_items)
    onehot = mlb.fit_transform(restricted)
    cooc = onehot.T @ onehot
    return pd.DataFrame(cooc, index=top_items, columns=top_items)


def missingness_report(df):
    """Reports n_missing and pct_missing per column, sorted by pct_missing
    descending."""
    return pd.DataFrame(
        {
            "n_missing": df.isnull().sum(),
            "pct_missing": df.isnull().mean() * 100,
        }
    ).sort_values("pct_missing", ascending=False)
