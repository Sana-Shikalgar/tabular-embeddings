"""Bundles every fitted per-type encoder plus cached SBERT vectors into
one object that loads and transforms data with a single call.
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
    """Every fitted per-type encoder for one candidate, plus the cached
    SBERT text vectors, dims for sizing each paradigm's embeddings, and
    a default list of columns eligible for SCARF-style corruption.
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
        encoder_dir: Path,
    ) -> "FittedEncoders":
        """Loads every fitted encoder and cached SBERT embedding for a
        candidate from encoder_dir, keyed by row id, without refitting.
        """
        missing_splits = [s for s in SPLITS if s not in split_dfs]
        assert not missing_splits, f"split_dfs missing required splits: {missing_splits}"
        assert ID_COL in train_split.id_cols, (
            f"{ID_COL!r} not in train_split.id_cols={train_split.id_cols}; "
            "the id-keyed text cache requires it"
        )

        train_df = split_dfs["train"]
        encoders_dir  = Path(encoder_dir)

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


    def transform(self, df: pd.DataFrame, include_text: bool = True) -> dict[str, np.ndarray]:
        """Transforms df into a dict of arrays, one per feature type
        ("numeric", "original_language", each list field, and --
        unless include_text=False -- "overview"/"original_title"). df
        must still carry `id`, used only to look up cached SBERT vectors.
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

        if include_text:
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


    def save(self, encoder_dir: Path) -> None:
        """Saves every sub-encoder and a dims/corruption_eligible_cols
        manifest to encoder_dir/. Does not re-save the SBERT
        .npy arrays, which fit() reads directly from that directory.
        """
        encoder_dir = Path(encoder_dir)

        save_json(self.standardizer, "standardizer.json", encoder_dir)
        save_json(self.ple, "ple.json", encoder_dir)
        save_json(
            self.language_lookup, "categorical_lookup_original_language.json", encoder_dir
        )
        for field, pooler in self.list_poolers.items():
            save_json(pooler, f"list_pooler_{field}.json", encoder_dir)

        encoders_dir = encoder_dir
        encoders_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = encoders_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(
                {"dims": self.dims, "corruption_eligible_cols": self.corruption_eligible_cols},
                f,
            )
        print(f"Saved: {manifest_path}")
