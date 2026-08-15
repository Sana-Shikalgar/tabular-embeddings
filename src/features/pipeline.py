"""Bundles the already-fitted Steps 3-8 encoders (Standardizer, PLE,
CategoricalLookup, ListFieldPoolers) plus the cached SBERT text vectors
into one object, so downstream code loads one thing instead of
re-deriving column groups and re-loading six JSON files and six .npy
files by hand.

SBERT vectors are looked up by row `id`, not row position. Section 3.4's
SCARF loop calls transform() twice per batch, every batch, so re-encoding
text live is off the table -- that's why Step 6 encoded it once, offline.
A positional cache would still be wrong, though: SGD reshuffles row order
between epochs, and Prompt 9.5's SCARF corruption leaves `id` intact
(text fields aren't corrupted by default) while corrupting other columns,
so `id` is the one thing guaranteed to still identify the row correctly.
A missing id is an error, not a live-encoding fallback -- this bundle
never loads the sentence-transformers model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.categorical import CategoricalLookup
from src.features.helper import load_json, save_json
from src.features.list_pooling import ListFieldPooler
from src.features.numeric_embeddings import PiecewiseLinearEncoder
from src.features.standardize import Standardizer
from src.features.target_split import ColumnGroups, FeatureTargetSplit, build_column_groups

TEXT_FIELDS = ("overview", "original_title")
SPLITS = ("train", "val", "test")
ID_COL = "id"


@dataclass(frozen=True)
class FittedEncoders:
    """Every fitted encoder Steps 3-8 produced for a candidate, loaded
    from ARTIFACTS_DIR/encoders/ rather than refit.

    language_lookup/list_poolers indices are shared across all three
    Section 3.4 paradigms (same vocab, same integers); each paradigm's
    trainable nn.Embedding table is its own, not part of this bundle.
    text_cache's SBERT vectors are the one thing that's frozen and
    literally shared as-is. text_cache is keyed by row `id`, not
    position -- see the module docstring for why; a missing id is an
    error, since this bundle holds no live TextEncoder to fall back to.

    column_groups.bypass_cols flows through the "numeric" key unchanged
    -- no lookup table, no further transform needed.

    dims is the per-transform()-key size a paradigm's model needs for
    sizing -- not always that key's array width. "numeric" is the real
    output width; "overview"/"original_title" are vector dims; but
    "original_language" and each list field are vocabulary sizes (for
    that paradigm's own nn.Embedding), not the index array's width.

    corruption_eligible_cols is a PLACEHOLDER for [D2] in Sections
    3.4/3.6 -- it does not decide a corruption rate or make the real
    eligible-column call (that belongs in Chapter 3's SCARF sub-section
    and risk register); it just gives that future decision one
    canonical list to live in. Defaults to every transform() key except
    "overview"/"original_title" (corrupting a frozen sentence embedding
    isn't meaningful the way corrupting a raw feature is, and Prompt
    9.3's caching depends on neither being corrupted). `id` isn't in
    this list because it was never a transform() output key.
    """

    standardizer: Standardizer
    ple: PiecewiseLinearEncoder
    language_lookup: CategoricalLookup
    list_poolers: dict[str, ListFieldPooler]
    text_cache: dict[str, dict[int, np.ndarray]]
    column_groups: ColumnGroups
    dims: dict[str, int]
    corruption_eligible_cols: list[str]

    @classmethod
    def fit(
        cls,
        split_dfs: dict[str, pd.DataFrame],
        train_split: FeatureTargetSplit,
        artifacts_dir: Path,
    ) -> "FittedEncoders":
        missing_splits = [s for s in SPLITS if s not in split_dfs]
        assert not missing_splits, f"split_dfs missing required splits: {missing_splits}"
        assert ID_COL in train_split.id_cols, (
            f"{ID_COL!r} not in train_split.id_cols={train_split.id_cols}; "
            "the id-keyed text cache requires it"
        )

        train_df = split_dfs["train"]
        artifacts_dir = Path(artifacts_dir)
        encoders_dir = artifacts_dir / "encoders"

        column_groups = build_column_groups(train_df, train_split)

        standardizer = load_json(Standardizer, encoders_dir / "standardizer.json")
        ple = load_json(PiecewiseLinearEncoder, encoders_dir / "ple.json")
        language_lookup = load_json(
            CategoricalLookup, encoders_dir / "categorical_lookup_original_language.json"
        )
        list_poolers = {
            field: load_json(ListFieldPooler, encoders_dir / f"list_pooler_{field}.json")
            for field in column_groups.list_cols
        }

        text_cache: dict[str, dict[int, np.ndarray]] = {}
        for field in TEXT_FIELDS:
            field_cache: dict[int, np.ndarray] = {}
            for split_name in SPLITS:
                path = encoders_dir / f"{field}_embeddings_{split_name}.npy"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing SBERT embedding file for field={field!r}, "
                        f"split={split_name!r}: {path}"
                    )
                embeddings = np.load(path)
                split_df = split_dfs[split_name]
                assert len(split_df) == len(embeddings), (
                    f"{field}/{split_name}: {len(split_df)} rows in split_dfs "
                    f"but {len(embeddings)} rows in the cached embeddings -- "
                    "they must be in the same row order for id-zipping to be correct"
                )

                ids = split_df[ID_COL].tolist()
                duplicate_ids = [i for i in ids if i in field_cache]
                assert not duplicate_ids, (
                    f"{field}: id(s) {duplicate_ids[:5]} already present in the "
                    "cache from an earlier split -- ids must be globally unique "
                    "across train/val/test for id-keyed lookup to be correct"
                )
                field_cache.update(zip(ids, embeddings))
            text_cache[field] = field_cache

        text_dims = {
            field: len(next(iter(text_cache[field].values())))
            for field in TEXT_FIELDS
        }
        assert text_dims["overview"] == text_dims["original_title"], (
            f"SBERT dim mismatch: overview={text_dims['overview']}, "
            f"original_title={text_dims['original_title']} -- both fields "
            "share one frozen encoder and should be identical"
        )

        dims = {
            "numeric": (
                sum(ple.n_bins_[c] for c in column_groups.numeric_cols)
                + len(column_groups.bypass_cols)
            ),
            "original_language": len(language_lookup.token_to_index),
            **text_dims,
            **{field: len(pooler.token_to_index) for field, pooler in list_poolers.items()},
        }

        corruption_eligible_cols = [key for key in dims if key not in TEXT_FIELDS]

        return cls(
            standardizer=standardizer,
            ple=ple,
            language_lookup=language_lookup,
            list_poolers=list_poolers,
            text_cache=text_cache,
            column_groups=column_groups,
            dims=dims,
            corruption_eligible_cols=corruption_eligible_cols,
        )


    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Transforms df (feature_cols only -- id_cols/target are never
        used as features) into a dict of arrays, one per paradigm key.
        df must still carry `id`, though: not a feature, just the cache
        key the text lookup below uses.

        "numeric": one float64 array, columns concatenated in a fixed
        order -- column_groups.numeric_cols first (each contributing
        ple.n_bins_[col] columns, in that column's bin order), THEN
        column_groups.bypass_cols (one raw passthrough column each). A
        caller slicing "numeric" back apart by original column (e.g. the
        FT-Transformer) walks numeric_cols summing ple.n_bins_[col],
        then appends one offset per bypass_cols entry.

        "original_language": int64 index array from language_lookup.

        "overview" / "original_title": looked up from text_cache by
        each row's `id`, in df's row order -- not re-encoded, not
        looked up by position (see module docstring for why: SGD
        reshuffling, SCARF leaving `id` intact). A missing id raises
        KeyError naming it -- no live encoder to fall back to.

        One key per column_groups.list_cols field: a dtype=object array
        of one variable-length Python list of int indices per row --
        deliberately ragged, not fixed-width; padding/packing is each
        paradigm's own responsibility.
        """
        assert ID_COL in df.columns, (
            f"transform() needs {ID_COL!r} present in df to look up cached "
            "SBERT vectors by id -- it is not used as a model feature, only "
            "as a cache key"
        )

        df_std = self.standardizer.transform(df)
        ple_result = self.ple.transform(df_std)

        numeric_parts = [ple_result[col] for col in self.column_groups.numeric_cols]
        numeric_parts.append(
            df[self.column_groups.bypass_cols].to_numpy(dtype=np.float64)
        )
        result: dict[str, np.ndarray] = {
            "numeric": np.concatenate(numeric_parts, axis=1),
            "original_language": self.language_lookup.transform(df),
        }

        row_ids = df[ID_COL].tolist()
        for field in TEXT_FIELDS:
            field_cache = self.text_cache[field]
            missing_ids = [i for i in row_ids if i not in field_cache]
            if missing_ids:
                raise KeyError(
                    f"{field}: {len(missing_ids)} row id(s) not found in the "
                    f"precomputed SBERT cache (first few: {missing_ids[:5]}) -- "
                    "transform() only looks up already-encoded rows, it never "
                    "re-encodes text live"
                )
            result[field] = np.stack([field_cache[i] for i in row_ids])

        for field, pooler in self.list_poolers.items():
            result[field] = pooler.transform(df).to_numpy()

        return result


    def save(self, artifacts_dir: Path) -> None:
        """Saves every sub-encoder to artifacts_dir/encoders/ under the
        exact filenames fit() reads back (standardizer.json, ple.json,
        categorical_lookup_original_language.json,
        list_pooler_{field}.json), via save_json. Steps 3-8 normally
        already saved these, so this is usually a no-op /
        overwrite-with-identical-content.

        Also writes dims/corruption_eligible_cols to a small
        manifest.json via plain json.dump (they're dict/list, not
        objects with to_dict()). Write-only: fit() recomputes both
        fresh rather than reading it back.

        Does NOT re-save the SBERT .npy arrays -- fit() reads those from
        artifacts_dir/encoders/ directly; duplicating multi-hundred-MB
        arrays into a second location would be wasteful for no benefit.
        """
        artifacts_dir = Path(artifacts_dir)

        save_json(self.standardizer, "standardizer.json", artifacts_dir)
        save_json(self.ple, "ple.json", artifacts_dir)
        save_json(
            self.language_lookup, "categorical_lookup_original_language.json", artifacts_dir
        )
        for field, pooler in self.list_poolers.items():
            save_json(pooler, f"list_pooler_{field}.json", artifacts_dir)

        encoders_dir = artifacts_dir / "encoders"
        encoders_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = encoders_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {"dims": self.dims, "corruption_eligible_cols": self.corruption_eligible_cols},
                f,
            )
        print(f"Saved: {manifest_path}")
